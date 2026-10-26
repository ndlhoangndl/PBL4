using System.Data;
using Microsoft.EntityFrameworkCore;
using Microsoft.Data.SqlClient;
using SmartDoor.Core;

namespace SmartDoor.Infrastructure;

public sealed class DoorDbContext(DbContextOptions<DoorDbContext> options):DbContext(options)
{
    public DbSet<User> Users=>Set<User>();
    public DbSet<OtpCode> OtpCodes=>Set<OtpCode>();
    public DbSet<AuthenticationSession> AuthenticationSessions=>Set<AuthenticationSession>();
    public DbSet<AccessEvent> AccessEvents=>Set<AccessEvent>();
    public DbSet<DoorCommand> DoorCommands=>Set<DoorCommand>();
    protected override void OnModelCreating(ModelBuilder model)
    {
        model.Entity<User>(b=>
        {
            b.HasKey(x=>x.Slot);
            b.Property(x=>x.Slot).ValueGeneratedNever();
            b.Property(x=>x.Name).HasMaxLength(100);
            b.ToTable(t=>t.HasCheckConstraint("CK_Users_Slot","[Slot] BETWEEN 1 AND 4"));
        });
        model.Entity<OtpCode>(b=>
        {
            b.HasKey(x=>x.Id);
            b.Property(x=>x.Digest).HasMaxLength(64).IsFixedLength();
            b.HasIndex(x=>x.Digest).IsUnique();
        });
        model.Entity<AuthenticationSession>().HasKey(x=>x.Id);
        model.Entity<AccessEvent>(b=>
        {
            b.HasKey(x=>x.Id);
            b.HasIndex(x=>x.TimestampUtc);
            b.Property(x=>x.Method).HasMaxLength(30);
            b.Property(x=>x.Result).HasMaxLength(30);
            b.HasOne<AuthenticationSession>().WithMany().HasForeignKey(x=>x.SessionId).OnDelete(DeleteBehavior.Restrict);
        });
        model.Entity<DoorCommand>(b=>
        {
            b.HasKey(x=>x.Id);
            b.Property(x=>x.Status).HasMaxLength(50);
            b.HasOne<AuthenticationSession>().WithMany().HasForeignKey(x=>x.SessionId).OnDelete(DeleteBehavior.Restrict);
        });
    }
}
public sealed class SqlStore(string connectionString):IStore
{
    public DoorDbContext Open()=>new(new DbContextOptionsBuilder<DoorDbContext>().UseSqlServer(connectionString,b=>b.CommandTimeout(10)).Options);
    public async Task InitializeAsync(CancellationToken ct)
    {
        await using var db=Open();
        await db.Database.EnsureCreatedAsync(ct);
        // Crash recovery never retries a potentially accepted door command.
        await db.AuthenticationSessions.Where(x=>x.State==SessionState.AwaitingFace||x.State==SessionState.ProcessingFace||x.State==SessionState.AwaitingOtp||x.State==SessionState.ProcessingOtp)
            .ExecuteUpdateAsync(x=>x.SetProperty(s=>s.State,SessionState.Cancelled).SetProperty(s=>s.Reason,"Phiên bị gián đoạn khi ứng dụng dừng"),ct);
        await db.DoorCommands.Where(x=>x.Status=="Sent"||x.Status=="Prepared").ExecuteUpdateAsync(x=>x.SetProperty(c=>c.Status,"UnknownAfterRestart"),ct);
    }
    public async Task<List<User>> UsersAsync(CancellationToken ct)
    {
        await using var db=Open();
        return await db.Users.AsNoTracking().OrderBy(x=>x.Slot).ToListAsync(ct);
    }
    public async Task SaveUserAsync(User user,CancellationToken ct)
    {
        await using var db=Open();
        var existing=await db.Users.FindAsync([user.Slot],ct);
        if(existing==null) db.Users.Add(user);
        else db.Entry(existing).CurrentValues.SetValues(user);
        await db.SaveChangesAsync(ct);
    }
    public async Task DeleteUserAsync(int slot,CancellationToken ct)
    {
        await using var db=Open();
        await db.Users.Where(x=>x.Slot==slot).ExecuteDeleteAsync(ct);
    }
    public async Task<bool> AddOtpAsync(OtpCode code,CancellationToken ct)
    {
        await using var db=Open();
        db.OtpCodes.Add(code);
        try
        {
            await db.SaveChangesAsync(ct);
            return true;
        }
        catch(DbUpdateException ex) when(ex.InnerException is SqlException { Number:2601 or 2627 }) { return false; }
    }
    public async Task<List<OtpCode>> OtpsAsync(CancellationToken ct)
    {
        await using var db=Open();
        return await db.OtpCodes.AsNoTracking().OrderByDescending(x=>x.CreatedUtc).Take(500).ToListAsync(ct);
    }
    public async Task RevokeOtpAsync(Guid id,CancellationToken ct)
    {
        await using var db=Open();
        await db.OtpCodes.Where(x=>x.Id==id&&x.ConsumedUtc==null).ExecuteUpdateAsync(x=>x.SetProperty(o=>o.Revoked,true),ct);
    }
    public async Task SaveSessionAsync(AuthenticationSession session,CancellationToken ct)
    {
        await using var db=Open();
        var existing=await db.AuthenticationSessions.FindAsync([session.Id],ct);
        if(existing==null) db.AuthenticationSessions.Add(session);
        else
        {
            // FailedAttempts belongs to the atomic verification transaction.
            session.FailedAttempts=Math.Max(session.FailedAttempts,existing.FailedAttempts);
            db.Entry(existing).CurrentValues.SetValues(session);
        }
        await db.SaveChangesAsync(ct);
    }
    public async Task<string> ConsumeOtpAsync(string digest,Guid session,DateTime now,int limit,CancellationToken ct)
    {
        await using var db=Open();
        await using var transaction=await db.Database.BeginTransactionAsync(IsolationLevel.Serializable,ct);
        // UPDLOCK removes reader-to-writer lock conversion races. Session then OTP order.
        var active=await db.AuthenticationSessions.FromSqlInterpolated($"SELECT * FROM AuthenticationSessions WITH (UPDLOCK,HOLDLOCK) WHERE Id={session}").SingleOrDefaultAsync(ct);
        if(active==null||active.State!=SessionState.ProcessingOtp||active.ExpiresUtc<=now||active.FailedAttempts>=limit) return "SessionRejected";
        var code=await db.OtpCodes.FromSqlInterpolated($"SELECT * FROM OtpCodes WITH (UPDLOCK,HOLDLOCK) WHERE Digest={digest}").SingleOrDefaultAsync(ct);
        now=DateTime.UtcNow; // Recheck expiry after any lock wait, rather than trusting request-start time.
        if(active.ExpiresUtc<=now) return "SessionRejected";
        var result=code==null?"Incorrect":code.ConsumedUtc!=null?"Consumed":code.Revoked?"Revoked":code.ExpiresUtc<=now?"Expired":"Accepted";
        if(result=="Accepted") code!.ConsumedUtc=now;
        else
        {
            active.FailedAttempts++;
            if(active.FailedAttempts>=limit) active.State=SessionState.Denied;
        }
        await db.SaveChangesAsync(ct);
        await transaction.CommitAsync(ct);
        return result;
    }
    public async Task<bool> ValidateUserAsync(int slot,long revision,AiMode mode,CancellationToken ct)
    {
        await using var db=Open();
        return await db.Users.AnyAsync(x=>x.Slot==slot&&x.Revision==revision&&x.Enabled&&x.Synchronized&&x.TemplateMode==mode,ct);
    }
    public async Task AddEventAsync(AccessEvent entry,CancellationToken ct)
    {
        await using var db=Open();
        db.AccessEvents.Add(entry);
        await db.SaveChangesAsync(ct);
    }
    public async Task SaveCommandAsync(DoorCommand command,CancellationToken ct)
    {
        await using var db=Open();
        var existing=await db.DoorCommands.FindAsync([command.Id],ct);
        if(existing==null) db.DoorCommands.Add(command);
        else db.Entry(existing).CurrentValues.SetValues(command);
        await db.SaveChangesAsync(ct);
    }
    public async Task<List<AccessEvent>> EventsAsync(DateTime fromUtc,DateTime toUtc,CancellationToken ct)
    {
        await using var db=Open();
        return await db.AccessEvents.AsNoTracking().Where(x=>x.TimestampUtc>=fromUtc&&x.TimestampUtc<=toUtc).OrderByDescending(x=>x.TimestampUtc).Take(1000).ToListAsync(ct);
    }
    public async Task<List<DoorCommand>> CommandsAsync(CancellationToken ct)
    {
        await using var db=Open();
        return await db.DoorCommands.AsNoTracking().OrderByDescending(x=>x.CreatedUtc).Take(1000).ToListAsync(ct);
    }
}
