using System.Text.Json;
using SmartDoor.Core;

namespace SmartDoor.Tests;

// Test double ONLY. The desktop always uses SQL Server, including Mock AI mode.
internal sealed class MemoryStore:IStore
{
    private readonly object gate=new();
    public List<User> Users { get; }=[];
    public List<OtpCode> Otps { get; }=[];
    public Dictionary<Guid,AuthenticationSession> Sessions { get; }=[];
    public List<AccessEvent> Events { get; }=[];
    public List<DoorCommand> Commands { get; }=[];
    private static T Copy<T>(T value)=>JsonSerializer.Deserialize<T>(JsonSerializer.Serialize(value))!;
    public Task InitializeAsync(CancellationToken ct)=>Task.CompletedTask;
    public Task<List<User>> UsersAsync(CancellationToken ct) { lock(gate) return Task.FromResult(Users.OrderBy(x=>x.Slot).Select(Copy).ToList()); }
    public Task SaveUserAsync(User user,CancellationToken ct) { lock(gate) { Users.RemoveAll(x=>x.Slot==user.Slot); Users.Add(Copy(user)); } return Task.CompletedTask; }
    public Task DeleteUserAsync(int slot,CancellationToken ct) { lock(gate) Users.RemoveAll(x=>x.Slot==slot); return Task.CompletedTask; }
    public Task<bool> AddOtpAsync(OtpCode code,CancellationToken ct)
    {
        lock(gate)
        {
            if(Otps.Any(x=>x.Digest==code.Digest)) return Task.FromResult(false);
            Otps.Add(Copy(code));
            return Task.FromResult(true);
        }
    }
    public Task<List<OtpCode>> OtpsAsync(CancellationToken ct) { lock(gate) return Task.FromResult(Otps.Select(Copy).ToList()); }
    public Task RevokeOtpAsync(Guid id,CancellationToken ct) { lock(gate) Otps.Single(x=>x.Id==id).Revoked=true; return Task.CompletedTask; }
    public Task SaveSessionAsync(AuthenticationSession session,CancellationToken ct)
    {
        lock(gate)
        {
            var copy=Copy(session);
            if(Sessions.TryGetValue(session.Id,out var old)) copy.FailedAttempts=Math.Max(old.FailedAttempts,copy.FailedAttempts);
            Sessions[session.Id]=copy;
        }
        return Task.CompletedTask;
    }
    public Task<string> ConsumeOtpAsync(string digest,Guid session,DateTime now,int limit,CancellationToken ct)
    {
        lock(gate)
        {
            var active=Sessions[session];
            if(active.State!=SessionState.ProcessingOtp||active.ExpiresUtc<=now||active.FailedAttempts>=limit) return Task.FromResult("SessionRejected");
            var code=Otps.SingleOrDefault(x=>x.Digest==digest);
            var result=code==null?"Incorrect":code.ConsumedUtc!=null?"Consumed":code.Revoked?"Revoked":code.ExpiresUtc<=now?"Expired":"Accepted";
            if(result=="Accepted") code!.ConsumedUtc=now;
            else active.FailedAttempts++;
            return Task.FromResult(result);
        }
    }
    public Task<bool> ValidateUserAsync(int slot,long revision,AiMode mode,CancellationToken ct) { lock(gate) return Task.FromResult(Users.Any(x=>x.Slot==slot&&x.Enabled&&x.Synchronized&&x.Revision==revision&&x.TemplateMode==mode)); }
    public Task AddEventAsync(AccessEvent entry,CancellationToken ct) { lock(gate) Events.Add(Copy(entry)); return Task.CompletedTask; }
    public Task SaveCommandAsync(DoorCommand command,CancellationToken ct) { lock(gate) { Commands.RemoveAll(x=>x.Id==command.Id); Commands.Add(Copy(command)); } return Task.CompletedTask; }
    public Task<List<AccessEvent>> EventsAsync(DateTime fromUtc,DateTime toUtc,CancellationToken ct) { lock(gate) return Task.FromResult(Events.Select(Copy).ToList()); }
    public Task<List<DoorCommand>> CommandsAsync(CancellationToken ct) { lock(gate) return Task.FromResult(Commands.Select(Copy).ToList()); }
}
