using System.Globalization;
using System.Security.Cryptography;
using System.Text;

namespace SmartDoor.Core;

public enum SessionState { Idle, AwaitingFace, ProcessingFace, AwaitingOtp, ProcessingOtp, Granted, Denied, Cancelled, TimedOut }
public enum AiMode { Real, Mock }
public enum OtpState { Active, Consumed, Expired, Revoked }
public sealed class AuthorizationGate
{
    public SemaphoreSlim Semaphore { get; }=new(1,1);
}
public sealed class User
{
    public int Slot { get; set; }
    public string Name { get; set; }="";
    public bool Enabled { get; set; }=true;
    public bool Synchronized { get; set; }
    public int Samples { get; set; }
    public AiMode TemplateMode { get; set; }
    public long Revision { get; set; }=BitConverter.ToInt64(RandomNumberGenerator.GetBytes(8))&long.MaxValue;
}
public sealed class OtpCode
{
    public Guid Id { get; set; }=Guid.NewGuid();
    public string Digest { get; set; }="";
    public DateTime CreatedUtc { get; set; }
    public DateTime ExpiresUtc { get; set; }
    public DateTime? ConsumedUtc { get; set; }
    public bool Revoked { get; set; }
    public OtpState State(DateTime now)=>ConsumedUtc!=null?OtpState.Consumed:Revoked?OtpState.Revoked:ExpiresUtc<=now?OtpState.Expired:OtpState.Active;
}
public sealed class AuthenticationSession
{
    public Guid Id { get; set; }=Guid.NewGuid();
    public DateTime StartedUtc { get; set; }
    public DateTime ExpiresUtc { get; set; }
    public SessionState State { get; set; }
    public AiMode Mode { get; set; }
    public int FailedAttempts { get; set; }
    public string Reason { get; set; }="";
}
public sealed class AccessEvent
{
    public Guid Id { get; set; }=Guid.NewGuid();
    public Guid SessionId { get; set; }
    public DateTime TimestampUtc { get; set; }
    public int? UserSlot { get; set; }
    public string Method { get; set; }="";
    public AiMode Mode { get; set; }
    public string Result { get; set; }="";
    public string Reason { get; set; }="";
    public Guid? CommandId { get; set; }
}
public sealed class DoorCommand
{
    public Guid Id { get; set; }=Guid.NewGuid();
    public Guid SessionId { get; set; }
    public DateTime CreatedUtc { get; set; }
    public string Status { get; set; }="Prepared";
    public bool Simulated { get; set; }=true;
}
public record AiUser(int Slot,string Name,int Samples);
public record FaceResult(string SessionId,string RequestId,long ElapsedMs,string Status,int? Slot,string Liveness,double? Similarity);
public record VoiceResult(string SessionId,string RequestId,long ElapsedMs,string Status,string? Digits);
public record EnrollmentResult(string SessionId,int Slot,int Samples,string Status);
public record DoorReply(string Status,bool Acknowledged);
public record DoorSnapshot(bool Connected,bool PersonPresent,bool Open,string BoltPosition,int OpenCount);

public interface IAiClient
{
    AiMode Mode { get; }
    Task<string> HealthAsync(CancellationToken ct);
    Task<IReadOnlyList<AiUser>> UsersAsync(CancellationToken ct);
    Task<EnrollmentResult> EnrollAsync(int slot,string name,IReadOnlyList<string> images,bool replace,Guid session,CancellationToken ct);
    Task DeleteAsync(int slot,CancellationToken ct);
    Task<FaceResult> RecognizeAsync(string file,Guid session,double threshold,double margin,CancellationToken ct);
    Task<VoiceResult> TranscribeAsync(string file,Guid session,CancellationToken ct);
}
public interface IDoorController
{
    DoorSnapshot Snapshot { get; }
    Task<DoorReply> SendOpenAsync(Guid commandId,Func<Func<DoorReply>,DoorReply> commitGuard,CancellationToken ct);
}
public interface IStore
{
    Task InitializeAsync(CancellationToken ct);
    Task<List<User>> UsersAsync(CancellationToken ct);
    Task SaveUserAsync(User user,CancellationToken ct);
    Task DeleteUserAsync(int slot,CancellationToken ct);
    Task<bool> AddOtpAsync(OtpCode code,CancellationToken ct);
    Task<List<OtpCode>> OtpsAsync(CancellationToken ct);
    Task RevokeOtpAsync(Guid id,CancellationToken ct);
    Task SaveSessionAsync(AuthenticationSession session,CancellationToken ct);
    Task<string> ConsumeOtpAsync(string digest,Guid session,DateTime now,int limit,CancellationToken ct);
    Task<bool> ValidateUserAsync(int slot,long revision,AiMode mode,CancellationToken ct);
    Task AddEventAsync(AccessEvent entry,CancellationToken ct);
    Task SaveCommandAsync(DoorCommand command,CancellationToken ct);
    Task<List<AccessEvent>> EventsAsync(DateTime fromUtc,DateTime toUtc,CancellationToken ct);
    Task<List<DoorCommand>> CommandsAsync(CancellationToken ct);
}
public sealed class OtpService(IStore store,byte[] key,TimeProvider clock)
{
    private readonly byte[] secret=key.Length>=32?key.ToArray():throw new ArgumentException("Khóa HMAC cần ít nhất 32 byte.");
    public static string Format(int number)=>number.ToString("D6",CultureInfo.InvariantCulture);
    public static bool IsValid(string? digits)=>digits is { Length:6 }&&digits.All(c=>c>='0'&&c<='9');
    public string Digest(string digits)=>Convert.ToHexString(HMACSHA256.HashData(secret,Encoding.UTF8.GetBytes(digits)));
    public async Task<(Guid Id,string Code)> CreateAsync(TimeSpan lifetime,CancellationToken ct=default)
    {
        if(lifetime<TimeSpan.FromSeconds(30)||lifetime>TimeSpan.FromHours(24)) throw new ArgumentException("Thời hạn OTP: 30 giây đến 24 giờ.");
        for(var attempt=0;attempt<32;attempt++)
        {
            var digits=Format(RandomNumberGenerator.GetInt32(1_000_000));
            var now=clock.GetUtcNow().UtcDateTime;
            var code=new OtpCode { Digest=Digest(digits),CreatedUtc=now,ExpiresUtc=now+lifetime };
            // Unique digests prevent reissuing any code still retained in the ledger.
            if(await store.AddOtpAsync(code,ct)) return (code.Id,digits);
        }
        throw new InvalidOperationException("Không sinh được OTP duy nhất; cần kiểm tra chính sách lưu lịch sử.");
    }
}
