using Microsoft.Data.SqlClient;
using Microsoft.EntityFrameworkCore;
using System.Security.Cryptography;
using SmartDoor.Core;
using SmartDoor.Infrastructure;
using Xunit;

namespace SmartDoor.Tests;

public sealed class SqlFactAttribute:FactAttribute
{
    public SqlFactAttribute()
    {
        if(string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("PBL4_TEST_SQL_BASE"))) Skip="Chưa cấu hình PBL4_TEST_SQL_BASE; không chạy SQL Server integration.";
    }
}
public sealed class SqlIntegrationTests
{
    private sealed class Database:IAsyncDisposable
    {
        private readonly string name="Pbl4SmartDoor_Test_"+Guid.NewGuid().ToString("N");
        public SqlStore Store { get; }
        public Database()
        {
            var builder=new SqlConnectionStringBuilder(Environment.GetEnvironmentVariable("PBL4_TEST_SQL_BASE")) { InitialCatalog=name };
            Store=new(builder.ConnectionString);
        }
        public async ValueTask DisposeAsync()
        {
            await using var db=Store.Open();
            // Only the freshly generated disposable test catalog is ever dropped.
            Assert.Equal(name,new SqlConnectionStringBuilder(db.Database.GetConnectionString()).InitialCatalog);
            Assert.StartsWith("Pbl4SmartDoor_Test_",name);
            await db.Database.EnsureDeletedAsync();
        }
    }
    [SqlFact]
    public async Task SqlConcurrentOtpConsumptionIsAtomic()
    {
        await using var database=new Database();
        var store=database.Store;
        await store.InitializeAsync(default);
        var otp=new OtpService(store,RandomNumberGenerator.GetBytes(32),TimeProvider.System);
        var created=await otp.CreateAsync(TimeSpan.FromMinutes(1));
        var sessions=Enumerable.Range(0,8).Select(_=>new AuthenticationSession { State=SessionState.ProcessingOtp,StartedUtc=DateTime.UtcNow,ExpiresUtc=DateTime.UtcNow.AddMinutes(1) }).ToArray();
        foreach(var session in sessions) await store.SaveSessionAsync(session,default);
        var results=await Task.WhenAll(sessions.Select(s=>store.ConsumeOtpAsync(otp.Digest(created.Code),s.Id,DateTime.UtcNow,3,default)));
        Assert.Single(results,r=>r=="Accepted");
        Assert.Equal(7,results.Count(r=>r=="Consumed"));
        Assert.NotNull((await store.OtpsAsync(default)).Single().ConsumedUtc);
    }
    [SqlFact]
    public async Task SqlBusinessFlowAndRecovery()
    {
        await using var database=new Database();
        var store=database.Store;
        await store.InitializeAsync(default);
        var gate=new AuthorizationGate();
        var ai=new MockAiClient();
        var users=new UserService(store,ai,gate);
        var otp=new OtpService(store,RandomNumberGenerator.GetBytes(32),TimeProvider.System);
        var door=new MockDoorController { Delay=TimeSpan.Zero };
        var engine=new AuthenticationEngine(store,ai,door,otp,TimeProvider.System,gate);
        await users.SaveAsync(1,"Integration tester",true);
        await users.EnrollAsync(1,["one","two","three"],false);
        await engine.StartAsync();
        await engine.FaceAsync("mock input");
        Assert.Equal(SessionState.Granted,engine.Snapshot!.State);
        await users.SaveAsync(1,"Integration tester",false);
        await engine.StartAsync();
        await engine.FaceAsync("mock input");
        Assert.Equal(SessionState.Denied,engine.Snapshot!.State);
        var code=await otp.CreateAsync(TimeSpan.FromMinutes(1));
        await engine.StartAsync();
        await engine.UseOtpAsync();
        await engine.TestOtpAsync(code.Code);
        Assert.Equal(SessionState.Granted,engine.Snapshot!.State);
        var entries=await store.EventsAsync(DateTime.UtcNow.AddMinutes(-1),DateTime.UtcNow.AddMinutes(1),default);
        Assert.Equal(3,entries.Count);
        Assert.DoesNotContain(code.Code,System.Text.Json.JsonSerializer.Serialize(entries));
        await engine.StartAsync();
        var abandoned=engine.Snapshot!.Id;
        await store.InitializeAsync(default);
        await using var db=store.Open();
        Assert.Equal(SessionState.Cancelled,(await db.AuthenticationSessions.FindAsync(abandoned))!.State);
        // Export the exact EF schema used by production, without creating any sample identities.
        var output=Environment.GetEnvironmentVariable("PBL4_SCHEMA_OUTPUT");
        if(!string.IsNullOrWhiteSpace(output)) await File.WriteAllTextAsync(output,db.Database.GenerateCreateScript());
    }
    [SqlFact]
    public async Task SqlOtpStatusesAndAttemptLimit()
    {
        await using var database=new Database();
        var store=database.Store;
        await store.InitializeAsync(default);
        var now=DateTime.UtcNow;
        var session=new AuthenticationSession { State=SessionState.ProcessingOtp,StartedUtc=now,ExpiresUtc=now.AddMinutes(1) };
        await store.SaveSessionAsync(session,default);
        await store.AddOtpAsync(new OtpCode { Digest=new string('A',64),CreatedUtc=now,ExpiresUtc=now.AddMinutes(-1) },default);
        await store.AddOtpAsync(new OtpCode { Digest=new string('B',64),CreatedUtc=now,ExpiresUtc=now.AddMinutes(1),Revoked=true },default);
        Assert.Equal("Expired",await store.ConsumeOtpAsync(new string('A',64),session.Id,now,3,default));
        Assert.Equal("Revoked",await store.ConsumeOtpAsync(new string('B',64),session.Id,now,3,default));
        Assert.Equal("Incorrect",await store.ConsumeOtpAsync(new string('C',64),session.Id,now,3,default));
        Assert.Equal("SessionRejected",await store.ConsumeOtpAsync(new string('D',64),session.Id,now,3,default));
    }
}
