namespace SmartDoor.Core;

public sealed class AuthenticationEngine(IStore store,IAiClient ai,IDoorController door,OtpService otp,TimeProvider clock,AuthorizationGate authorization)
{
    private readonly object stateLock=new();
    private readonly SemaphoreSlim operations=new(1,1);
    private AuthenticationSession? current;
    private CancellationTokenSource? lifetime;
    public double Threshold { get; set; }=0.363;
    public double Margin { get; set; }=0.05;
    public TimeSpan SessionLifetime { get; set; }=TimeSpan.FromMinutes(2);
    public int AttemptLimit { get; set; }=3;
    public AuthenticationSession? Snapshot
    {
        get
        {
            lock(stateLock)
                return current==null?null:new AuthenticationSession { Id=current.Id,StartedUtc=current.StartedUtc,ExpiresUtc=current.ExpiresUtc,State=current.State,Mode=current.Mode,Reason=current.Reason,FailedAttempts=current.FailedAttempts };
        }
    }
    private bool Active(Guid id)=>current?.Id==id&&(current.State is SessionState.AwaitingFace or SessionState.ProcessingFace or SessionState.AwaitingOtp or SessionState.ProcessingOtp)&&current.ExpiresUtc>clock.GetUtcNow().UtcDateTime&&lifetime?.IsCancellationRequested==false;
    public async Task StartAsync(CancellationToken ct=default)
    {
        if(!await operations.WaitAsync(0,ct)) throw new InvalidOperationException("Đang xử lý phiên trước.");
        try
        {
            lock(stateLock)
            {
                if(current!=null&&Active(current.Id)) throw new InvalidOperationException("Chỉ một phiên tại một thời điểm.");
                lifetime?.Dispose();
                lifetime=new CancellationTokenSource(SessionLifetime,clock);
                var now=clock.GetUtcNow().UtcDateTime;
                current=new AuthenticationSession { StartedUtc=now,ExpiresUtc=now+SessionLifetime,State=SessionState.AwaitingFace,Mode=ai.Mode };
            }
            await PersistAsync(ct);
        }
        finally { operations.Release(); }
    }
    public async Task CancelAsync(bool timeout=false)
    {
        lock(stateLock)
        {
            if(current==null||current.State is SessionState.Granted or SessionState.Denied or SessionState.Cancelled or SessionState.TimedOut) return;
            lifetime?.Cancel();
            current.State=timeout?SessionState.TimedOut:SessionState.Cancelled;
            current.Reason=timeout?"Phiên hết hạn":"Đã hủy phiên";
        }
        await operations.WaitAsync();
        try
        {
            await PersistAsync();
            var snapshot=Snapshot!;
            await LogAsync(snapshot.Id,"Session",snapshot.State.ToString(),snapshot.Reason,null,null);
        }
        finally { operations.Release(); }
    }
    public async Task TickAsync()
    {
        var snapshot=Snapshot;
        if(snapshot!=null&&snapshot.ExpiresUtc<=clock.GetUtcNow().UtcDateTime) await CancelAsync(true);
    }
    public async Task UseOtpAsync()
    {
        if(!await operations.WaitAsync(0)) throw new InvalidOperationException("Đang xử lý dữ liệu; hãy hủy hoặc đợi.");
        try
        {
            lock(stateLock)
            {
                if(current==null||!Active(current.Id)||current.State!=SessionState.AwaitingFace) throw new InvalidOperationException("Phiên chưa sẵn sàng.");
                current.State=SessionState.AwaitingOtp;
                current.Reason="Đọc mã OTP từ thiết bị ngoài hoặc nhập đầu vào kiểm thử";
            }
            await PersistAsync();
        }
        finally { operations.Release(); }
    }
    public Task FaceAsync(string file)=>ProcessAsync(true,file,false);
    public Task AudioAsync(string file)=>ProcessAsync(false,file,false);
    public Task TestOtpAsync(string digits)=>ProcessAsync(false,digits,true);
    private async Task ProcessAsync(bool face,string input,bool direct)
    {
        if(!await operations.WaitAsync(0)) throw new InvalidOperationException("Một yêu cầu đang xử lý.");
        Guid id=Guid.Empty;
        var method=face?"Face":direct?"TestOtp":"VoiceOtp";
        CancellationToken ct=default;
        var userLease=false;
        try
        {
            lock(stateLock)
            {
                if(current==null||!Active(current.Id)||current.State!=(face?SessionState.AwaitingFace:SessionState.AwaitingOtp)) throw new InvalidOperationException("Trạng thái phiên không phù hợp.");
                id=current.Id;
                current.State=face?SessionState.ProcessingFace:SessionState.ProcessingOtp;
                ct=lifetime!.Token;
            }
            await PersistAsync(ct);
            int? slot=null;
            if(face)
            {
                var before=await store.UsersAsync(ct);
                var result=await ai.RecognizeAsync(input,id,Threshold,Margin,ct);
                EnsureCurrent(id,result.SessionId);
                if(result.Status!="candidate"||result.Slot==null)
                {
                    await RetryAsync(id,SessionState.AwaitingOtp,"Người lạ; chuyển sang OTP",method,ct);
                    return;
                }
                if(ai.Mode==AiMode.Real||result.Liveness!="passed")
                {
                    await RetryAsync(id,SessionState.AwaitingOtp,$"Ứng viên vị trí {result.Slot}; chưa có chống giả mạo. Cần OTP.",method,ct);
                    return;
                }
                var candidate=before.SingleOrDefault(x=>x.Slot==result.Slot);
                await authorization.Semaphore.WaitAsync(ct);
                userLease=true;
                var after=await store.UsersAsync(ct);
                var remote=await ai.UsersAsync(ct);
                if(!before.Select(x=>(x.Slot,x.Revision)).SequenceEqual(after.Select(x=>(x.Slot,x.Revision))))
                    throw new InvalidOperationException("Danh tính hoặc mẫu đã thay đổi trong lúc xử lý.");
                if(candidate==null||!candidate.Enabled||!candidate.Synchronized||candidate.TemplateMode!=ai.Mode||!remote.Any(x=>x.Slot==candidate.Slot&&x.Name==candidate.Name&&x.Samples==candidate.Samples)||!await store.ValidateUserAsync(candidate.Slot,candidate.Revision,ai.Mode,ct))
                    throw new InvalidOperationException("Người dùng bị vô hiệu hóa hoặc mẫu chưa đồng bộ.");
                slot=candidate.Slot;
            }
            else
            {
                string? digits=input;
                if(!direct)
                {
                    var result=await ai.TranscribeAsync(input,id,ct);
                    EnsureCurrent(id,result.SessionId);
                    digits=result.Status=="decoded"?result.Digits:null;
                }
                EnsureCurrent(id,id.ToString());
                // Invalid length still counts as ONE submitted OTP attempt. Never log input.
                var outcome=await store.ConsumeOtpAsync(OtpService.IsValid(digits)?otp.Digest(digits!):"invalid",id,clock.GetUtcNow().UtcDateTime,AttemptLimit,ct);
                if(outcome!="Accepted")
                {
                    lock(stateLock)
                    {
                        if(!Active(id)) throw new OperationCanceledException();
                        current!.FailedAttempts++;
                    }
                    if(Snapshot!.FailedAttempts>=AttemptLimit) await FinishAsync(id,SessionState.Denied,"Đã đạt giới hạn thử OTP",method,null,null);
                    else await RetryAsync(id,SessionState.AwaitingOtp,$"OTP: {outcome}",method,ct);
                    return;
                }
            }
            EnsureCurrent(id,id.ToString());
            var command=new DoorCommand { SessionId=id,CreatedUtc=clock.GetUtcNow().UtcDateTime,Status="Sent" };
            await store.SaveCommandAsync(command,ct);
            DoorReply reply;
            try
            {
                reply=await door.SendOpenAsync(command.Id,action=>
                {
                    lock(stateLock)
                    {
                        if(!Active(id)) throw new OperationCanceledException("Phiên đã hủy/hết hạn.");
                        var result=action(); // Commit simulation atomically against cancellation.
                        if(result.Acknowledged) current!.State=SessionState.Granted;
                        return result;
                    }
                },ct);
            }
            catch(OperationCanceledException)
            {
                command.Status="CancelledOrTimedOut";
                await store.SaveCommandAsync(command,CancellationToken.None);
                throw;
            }
            command.Status=reply.Status;
            await store.SaveCommandAsync(command,CancellationToken.None);
            await FinishAsync(id,reply.Acknowledged?SessionState.Granted:SessionState.Denied,reply.Acknowledged?"Bộ mô phỏng đã xác nhận mở cửa":"Xác thực được chấp nhận nhưng lệnh cửa mô phỏng thất bại; OTP không được khôi phục",method,slot,command.Id);
        }
        catch(OperationCanceledException)
        {
            lock(stateLock)
            {
                if(current?.Id==id&&current.State!=SessionState.Granted)
                {
                    current.State=current.State==SessionState.Cancelled?SessionState.Cancelled:SessionState.TimedOut;
                    current.Reason=current.State==SessionState.Cancelled?"Đã hủy; bỏ kết quả đến muộn":"Hết thời gian; không phát sinh lệnh mở muộn";
                }
            }
            await PersistAsync();
            if(id!=Guid.Empty) await LogAsync(id,method,"Discarded","Kết quả phiên đã hủy/hết hạn",null,null);
        }
        catch(Exception ex) when(id!=Guid.Empty)
        {
            // Only controlled exception messages may be displayed; API body/transcript never logged.
            var reason=ex is InvalidOperationException?ex.Message:ex is AiInputException?ex.Message:ex is TimeoutException?"Dịch vụ AI timeout":ex is HttpRequestException?"Dịch vụ AI không sẵn sàng":"Lỗi database/dịch vụ; kiểm tra cấu hình";
            lock(stateLock)
            {
                if(Active(id))
                {
                    current!.State=SessionState.Denied;
                    current.Reason=reason;
                }
            }
            await PersistAsync();
            await LogAsync(id,method,"Failed",reason,null,null);
        }
        finally
        {
            if(userLease) authorization.Semaphore.Release();
            operations.Release();
        }
    }
    private void EnsureCurrent(Guid id,string responseSession)
    {
        lock(stateLock)
            if(!Active(id)||responseSession!=id.ToString()) throw new OperationCanceledException("Bỏ kết quả không thuộc phiên hiện tại.");
    }
    private async Task RetryAsync(Guid id,SessionState state,string reason,string method,CancellationToken ct)
    {
        lock(stateLock)
        {
            if(!Active(id)) throw new OperationCanceledException();
            current!.State=state;
            current.Reason=reason;
        }
        await PersistAsync(ct);
        await LogAsync(id,method,"Retry",reason,null,null);
    }
    private async Task FinishAsync(Guid id,SessionState state,string reason,string method,int? slot,Guid? command)
    {
        lock(stateLock)
        {
            if(current?.Id!=id) return;
            if(current.State is SessionState.Cancelled or SessionState.TimedOut) return;
            current.State=state;
            current.Reason=reason;
        }
        await PersistAsync();
        await LogAsync(id,method,state.ToString(),reason,slot,command);
    }
    private Task PersistAsync(CancellationToken ct=default)=>Snapshot is { } snapshot?store.SaveSessionAsync(snapshot,ct):Task.CompletedTask;
    private Task LogAsync(Guid id,string method,string result,string reason,int? slot,Guid? command)=>store.AddEventAsync(new AccessEvent { SessionId=id,TimestampUtc=clock.GetUtcNow().UtcDateTime,Mode=ai.Mode,Method=method,Result=result,Reason=reason,UserSlot=slot,CommandId=command },CancellationToken.None);
}
public sealed class AiInputException(string message):Exception(message);
