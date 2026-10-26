using System.Security.Cryptography;
using System.Text.Json;
using System.Text.Json.Serialization;
using SmartDoor.Core;
using SmartDoor.Infrastructure;

namespace SmartDoor.Desktop;

public sealed class AppSettings
{
    public AiMode AiMode { get; set; }=AiMode.Mock;
    public string AiUrl { get; set; }="http://127.0.0.1:8000";
    public string ConnectionString { get; set; }="";
    public int TimeoutSeconds { get; set; }=60;
    public double Threshold { get; set; }=0.363;
    public double Margin { get; set; }=0.05;
    public int SessionSeconds { get; set; }=120;
    private static readonly JsonSerializerOptions Json=new() { WriteIndented=true,Converters={ new JsonStringEnumConverter() } };
    private static string LocalPath=>Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),"Pbl4SmartDoor","appsettings.local.json");
    public static AppSettings Load()
    {
        var path=File.Exists(LocalPath)?LocalPath:Path.Combine(AppContext.BaseDirectory,"appsettings.json");
        var settings=JsonSerializer.Deserialize<AppSettings>(File.ReadAllText(path),Json)??throw new InvalidOperationException("Thiếu cấu hình.");
        settings.ConnectionString=Environment.GetEnvironmentVariable("PBL4_SQL_CONNECTION")??settings.ConnectionString;
        return settings;
    }
    public void Save()
    {
        if(!Uri.TryCreate(AiUrl,UriKind.Absolute,out var uri)||uri.Scheme is not ("http" or "https")||uri.AbsolutePath!="/") throw new ArgumentException("URL AI phải là địa chỉ gốc HTTP/HTTPS.");
        if(TimeoutSeconds is <1 or >300||SessionSeconds is <5 or >900||Threshold is <0 or >1||Margin is <0 or >1) throw new ArgumentException("Cấu hình thời gian/ngưỡng không hợp lệ.");
        Directory.CreateDirectory(Path.GetDirectoryName(LocalPath)!);
        File.WriteAllText(LocalPath,JsonSerializer.Serialize(this,Json));
    }
}
public static class SecretKey
{
    public static byte[] Load()
    {
        var environment=Environment.GetEnvironmentVariable("PBL4_OTP_HMAC_KEY");
        if(!string.IsNullOrWhiteSpace(environment))
        {
            var key=Convert.FromBase64String(environment);
            if(key.Length<32) throw new InvalidOperationException("PBL4_OTP_HMAC_KEY cần ít nhất 32 byte (Base64).");
            return key;
        }
        var folder=Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),"Pbl4SmartDoor");
        var path=Path.Combine(folder,"otp-hmac.dpapi");
        Directory.CreateDirectory(folder);
        if(!File.Exists(path))
        {
            var key=RandomNumberGenerator.GetBytes(32);
            // Current Windows user DPAPI; raw key is never written to source/config/database.
            var protectedKey=ProtectedData.Protect(key,null,DataProtectionScope.CurrentUser);
            try
            {
                using var file=new FileStream(path,FileMode.CreateNew,FileAccess.Write,FileShare.None);
                file.Write(protectedKey);
            }
            catch(IOException) when(File.Exists(path)) { }
            CryptographicOperations.ZeroMemory(key);
        }
        return ProtectedData.Unprotect(File.ReadAllBytes(path),null,DataProtectionScope.CurrentUser);
    }
}
public sealed class AppController:IDisposable
{
    private readonly SqlStore store;
    private readonly HttpClient? http;
    private readonly IAiClient ai;
    private readonly UserService users;
    private readonly OtpService otp;
    private bool initialized;
    public AppSettings Settings { get; }
    public AuthenticationEngine Engine { get; }
    public MockDoorController Door { get; }=new();
    public MockAiClient? Mock=>ai as MockAiClient;
    public bool DatabaseReady { get; private set; }
    public bool SimulateDatabaseFailure { get; set; }
    public string DatabaseStatus { get; private set; }="Chưa kiểm tra";
    public string AiStatus { get; private set; }="Chưa kiểm tra";
    public AppController(AppSettings settings)
    {
        Settings=settings;
        store=new(settings.ConnectionString);
        if(settings.AiMode==AiMode.Mock) ai=new MockAiClient();
        else
        {
            http=new HttpClient { BaseAddress=new Uri(settings.AiUrl),Timeout=TimeSpan.FromSeconds(settings.TimeoutSeconds) };
            ai=new RealAiClient(http);
        }
        var key=SecretKey.Load();
        otp=new(store,key,TimeProvider.System);
        CryptographicOperations.ZeroMemory(key);
        var gate=new AuthorizationGate();
        users=new(store,ai,gate);
        Engine=new(store,ai,Door,otp,TimeProvider.System,gate) { Threshold=settings.Threshold,Margin=settings.Margin,SessionLifetime=TimeSpan.FromSeconds(settings.SessionSeconds) };
    }
    public async Task CheckAsync()
    {
        using var timeout=new CancellationTokenSource(TimeSpan.FromSeconds(15));
        try
        {
            EnsureNotSimulatedFailure();
            if(!initialized)
            {
                await store.InitializeAsync(timeout.Token);
                initialized=true;
            }
            else await store.UsersAsync(timeout.Token);
            DatabaseReady=true;
            DatabaseStatus="SQL Server sẵn sàng";
        }
        catch(Exception)
        {
            DatabaseReady=false;
            DatabaseStatus=SimulateDatabaseFailure?"Mô phỏng DB không sẵn sàng":"Không kết nối được SQL Server. Kiểm tra instance, quyền tạo DB và connection string ở Cấu hình.";
        }
        try
        {
            AiStatus=await ai.HealthAsync(timeout.Token);
            if(DatabaseReady) await users.ReconcileAsync(timeout.Token);
        }
        catch(Exception) { AiStatus="AI không sẵn sàng/không kiểm tra được đồng bộ. Không tự chuyển chế độ."; }
    }
    private void EnsureNotSimulatedFailure()
    {
        if(SimulateDatabaseFailure) throw new InvalidOperationException("Mô phỏng database không sẵn sàng.");
    }
    private void Ready()
    {
        EnsureNotSimulatedFailure();
        if(!DatabaseReady) throw new InvalidOperationException(DatabaseStatus);
    }
    public Task StartAsync() { Ready(); return Engine.StartAsync(); }
    public Task FaceAsync(string path) { Ready(); return Engine.FaceAsync(path); }
    public Task AudioAsync(string path) { Ready(); return Engine.AudioAsync(path); }
    public Task TestOtpAsync(string input) { Ready(); return Engine.TestOtpAsync(input); }
    public Task<List<User>> UsersAsync() { Ready(); return store.UsersAsync(default); }
    public Task SaveUserAsync(int slot,string name,bool enabled) { Ready(); return users.SaveAsync(slot,name,enabled); }
    public Task EnrollAsync(int slot,IReadOnlyList<string> paths,bool replace) { Ready(); return users.EnrollAsync(slot,paths,replace); }
    public Task DeleteUserAsync(int slot) { Ready(); return users.DeleteAsync(slot); }
    public Task<(Guid Id,string Code)> CreateOtpAsync(int minutes) { Ready(); return otp.CreateAsync(TimeSpan.FromMinutes(minutes)); }
    public Task RevokeOtpAsync(Guid id) { Ready(); return store.RevokeOtpAsync(id,default); }
    public async Task<object[]> OtpRowsAsync()
    {
        Ready();
        return (await store.OtpsAsync(default)).Select(x=>(object)new { x.Id,TrangThai=x.State(DateTime.UtcNow).ToString(),TaoLuc=x.CreatedUtc.ToLocalTime(),HetHan=x.ExpiresUtc.ToLocalTime(),DaDung=x.ConsumedUtc?.ToLocalTime() }).ToArray();
    }
    public async Task<object[]> HistoryAsync(DateTime from,DateTime to,string method,string result)
    {
        Ready();
        var events=await store.EventsAsync(from.ToUniversalTime(),to.ToUniversalTime(),default);
        var commands=await store.CommandsAsync(default);
        return events.Where(x=>(method=="Tất cả"||x.Method==method)&&(result=="Tất cả"||x.Result==result)).Select(x=>(object)new { Luc=x.TimestampUtc.ToLocalTime(),Phien=x.SessionId,NguoiDung=x.UserSlot,PhuongThuc=x.Method,Nguon=x.Method=="TestOtp"?"Đầu vào kiểm thử":x.Method=="Exit"?"EXIT mô phỏng":x.Mode==AiMode.Mock?"Mock AI":"AI thật",KetQua=x.Result,LyDo=x.Reason,LenhCua=x.CommandId,TrangThaiLenh=commands.FirstOrDefault(c=>c.Id==x.CommandId)?.Status??"Không gửi",ThietBi="Mô phỏng" }).ToArray();
    }
    public async Task ExitAsync()
    {
        Ready();
        await Engine.CancelAsync();
        // Independent of AI; EXIT still requires audit database in this software simulator.
        var now=DateTime.UtcNow;
        var session=new AuthenticationSession { State=SessionState.Granted,StartedUtc=now,ExpiresUtc=now,Mode=Settings.AiMode,Reason="Nút EXIT mô phỏng" };
        await store.SaveSessionAsync(session,default);
        var command=new DoorCommand { SessionId=session.Id,CreatedUtc=now,Status="Sent" };
        await store.SaveCommandAsync(command,default);
        using var timeout=new CancellationTokenSource(TimeSpan.FromSeconds(4));
        var reply=await Door.SendOpenAsync(command.Id,action=>action(),timeout.Token);
        command.Status=reply.Status;
        await store.SaveCommandAsync(command,default);
        await store.AddEventAsync(new AccessEvent { SessionId=session.Id,TimestampUtc=now,Method="Exit",Mode=Settings.AiMode,Result=reply.Acknowledged?"Granted":"Failed",Reason="Nút EXIT; bộ mô phỏng cửa",CommandId=command.Id },default);
    }
    public void Dispose() { http?.Dispose(); }
}
