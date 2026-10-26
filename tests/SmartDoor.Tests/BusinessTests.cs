using System.Security.Cryptography;
using SmartDoor.Core;
using SmartDoor.Infrastructure;
using Xunit;

namespace SmartDoor.Tests;

public sealed class BusinessTests
{
    private sealed class Rig
    {
        public MemoryStore Store { get; }=new();
        public MockAiClient Ai { get; }=new();
        public MockDoorController Door { get; }=new() { Delay=TimeSpan.Zero };
        public AuthorizationGate Gate { get; }=new();
        public OtpService Otp { get; }
        public AuthenticationEngine Engine { get; }
        public UserService Users { get; }
        public Rig()
        {
            Otp=new(Store,RandomNumberGenerator.GetBytes(32),TimeProvider.System);
            Engine=new(Store,Ai,Door,Otp,TimeProvider.System,Gate);
            Users=new(Store,Ai,Gate);
        }
        public async Task EnrollAsync()
        {
            await Users.SaveAsync(1,"Người kiểm thử",true);
            await Users.EnrollAsync(1,["a","b","c"],false);
        }
        public async Task<(Guid Id,string Code)> OtpSessionAsync()
        {
            var code=await Otp.CreateAsync(TimeSpan.FromMinutes(1));
            await Engine.StartAsync();
            await Engine.UseOtpAsync();
            return code;
        }
    }
    [Fact]
    public void LeadingZeroAndInputValidation()
    {
        Assert.Equal("000007",OtpService.Format(7));
        Assert.True(OtpService.IsValid("012345"));
        Assert.False(OtpService.IsValid("12345"));
        Assert.False(OtpService.IsValid("１２３４５６"));
    }
    [Fact]
    public async Task CorrectOtpIsConsumedAndNeverLogged()
    {
        var rig=new Rig();
        var code=await rig.OtpSessionAsync();
        await rig.Engine.TestOtpAsync(code.Code);
        Assert.Equal(SessionState.Granted,rig.Engine.Snapshot!.State);
        Assert.Equal(1,rig.Door.Snapshot.OpenCount);
        Assert.NotNull(rig.Store.Otps.Single().ConsumedUtc);
        Assert.DoesNotContain(code.Code,System.Text.Json.JsonSerializer.Serialize(rig.Store.Events));
        Assert.DoesNotContain(code.Code,System.Text.Json.JsonSerializer.Serialize(rig.Store.Otps));
        await rig.Engine.StartAsync();
        await rig.Engine.UseOtpAsync();
        await rig.Engine.TestOtpAsync(code.Code);
        Assert.Contains("Consumed",rig.Engine.Snapshot!.Reason);
        Assert.Equal(1,rig.Door.Snapshot.OpenCount);
    }
    [Theory]
    [InlineData("Incorrect")]
    [InlineData("Expired")]
    [InlineData("Revoked")]
    public async Task OtpFailureIsExplained(string state)
    {
        var rig=new Rig();
        var code=await rig.OtpSessionAsync();
        if(state=="Expired") rig.Store.Otps.Single().ExpiresUtc=DateTime.UtcNow.AddSeconds(-1);
        if(state=="Revoked") await rig.Store.RevokeOtpAsync(code.Id,default);
        await rig.Engine.TestOtpAsync(state=="Incorrect"?"bad":code.Code);
        Assert.Contains(state,rig.Engine.Snapshot!.Reason);
        Assert.Equal(0,rig.Door.Snapshot.OpenCount);
    }
    [Fact]
    public async Task AttemptLimitDeniesSession()
    {
        var rig=new Rig();
        await rig.OtpSessionAsync();
        for(var i=0;i<3;i++) await rig.Engine.TestOtpAsync("bad");
        Assert.Equal(SessionState.Denied,rig.Engine.Snapshot!.State);
        Assert.Equal(3,rig.Engine.Snapshot.FailedAttempts);
    }
    [Fact]
    public async Task ConcurrentConsumersOnlyOneAccepted()
    {
        var rig=new Rig();
        var code=await rig.Otp.CreateAsync(TimeSpan.FromMinutes(1));
        var sessions=Enumerable.Range(0,2).Select(_=>new AuthenticationSession { State=SessionState.ProcessingOtp,ExpiresUtc=DateTime.UtcNow.AddMinutes(1) }).ToArray();
        foreach(var session in sessions) await rig.Store.SaveSessionAsync(session,default);
        var results=await Task.WhenAll(sessions.Select(s=>Task.Run(()=>rig.Store.ConsumeOtpAsync(rig.Otp.Digest(code.Code),s.Id,DateTime.UtcNow,3,default))));
        Assert.Single(results,x=>x=="Accepted");
        Assert.Single(results,x=>x=="Consumed");
    }
    [Fact]
    public async Task DisabledUserCannotOpen()
    {
        var rig=new Rig();
        await rig.EnrollAsync();
        await rig.Users.SaveAsync(1,"Người kiểm thử",false);
        await rig.Engine.StartAsync();
        await rig.Engine.FaceAsync("input");
        Assert.Equal(SessionState.Denied,rig.Engine.Snapshot!.State);
        Assert.Equal(0,rig.Door.Snapshot.OpenCount);
    }
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task LateAiAfterCancellationOrExpiryIsDiscarded(bool expiry)
    {
        var rig=new Rig();
        await rig.EnrollAsync();
        rig.Ai.Scenario=MockScenario.Delayed;
        rig.Ai.IgnoreCancellation=true;
        rig.Ai.Delay=TimeSpan.FromMilliseconds(180);
        if(expiry) rig.Engine.SessionLifetime=TimeSpan.FromMilliseconds(50);
        await rig.Engine.StartAsync();
        var work=rig.Engine.FaceAsync("input");
        if(!expiry)
        {
            await Task.Delay(30);
            await rig.Engine.CancelAsync();
        }
        await work;
        Assert.Equal(expiry?SessionState.TimedOut:SessionState.Cancelled,rig.Engine.Snapshot!.State);
        Assert.Equal(0,rig.Door.Snapshot.OpenCount);
        Assert.Empty(rig.Store.Commands);
    }
    [Fact]
    public async Task UserChangedDuringInferenceCannotOpen()
    {
        var rig=new Rig();
        await rig.EnrollAsync();
        rig.Ai.Scenario=MockScenario.Delayed;
        rig.Ai.Delay=TimeSpan.FromMilliseconds(150);
        await rig.Engine.StartAsync();
        var work=rig.Engine.FaceAsync("input");
        await Task.Delay(30);
        await rig.Users.SaveAsync(1,"Tên đã đổi",true);
        await work;
        Assert.Equal(SessionState.Denied,rig.Engine.Snapshot!.State);
        Assert.Equal(0,rig.Door.Snapshot.OpenCount);
    }
    [Fact]
    public async Task TemplatesReplacedDuringInferenceCannotOpen()
    {
        var rig=new Rig();
        await rig.EnrollAsync();
        rig.Ai.Scenario=MockScenario.Delayed;
        rig.Ai.Delay=TimeSpan.FromMilliseconds(150);
        await rig.Engine.StartAsync();
        var work=rig.Engine.FaceAsync("input");
        await Task.Delay(30);
        await rig.Users.EnrollAsync(1,["replacement1","replacement2","replacement3"],true);
        await work;
        Assert.Equal(SessionState.Denied,rig.Engine.Snapshot!.State);
        Assert.Equal(0,rig.Door.Snapshot.OpenCount);
    }
    [Fact]
    public async Task DuplicateCommandDoesNotOpenTwice()
    {
        var door=new MockDoorController { Delay=TimeSpan.FromMilliseconds(10) };
        var id=Guid.NewGuid();
        var replies=await Task.WhenAll(Enumerable.Range(0,10).Select(_=>door.SendOpenAsync(id,action=>action(),default)));
        Assert.All(replies,r=>Assert.True(r.Acknowledged));
        Assert.Equal(1,door.Snapshot.OpenCount);
    }
    [Fact]
    public async Task FailedEnrollmentStaysUnsynchronized()
    {
        var rig=new Rig();
        await rig.EnrollAsync();
        rig.Ai.Scenario=MockScenario.EnrollmentFailure;
        await Assert.ThrowsAsync<AiInputException>(()=>rig.Users.EnrollAsync(1,["d","e","f"],true));
        Assert.False(rig.Store.Users.Single().Synchronized);
    }
    [Fact]
    public async Task PartialDeletionDisablesUserAndKeepsUnsynchronizedRecord()
    {
        var rig=new Rig();
        await rig.EnrollAsync();
        rig.Ai.Scenario=MockScenario.DeleteFailure;
        await Assert.ThrowsAsync<AiInputException>(()=>rig.Users.DeleteAsync(1));
        var retained=rig.Store.Users.Single();
        Assert.False(retained.Enabled);
        Assert.False(retained.Synchronized);
        Assert.Single(await rig.Ai.UsersAsync(default));
    }
    [Fact]
    public async Task FifthSlotIsRejectedWithoutPersistingUser()
    {
        var rig=new Rig();
        await Assert.ThrowsAsync<ArgumentException>(()=>rig.Users.SaveAsync(5,"Test",true));
        Assert.Empty(rig.Store.Users);
    }
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task DoorDisconnectedOrNoResponseKeepsOtpConsumed(bool noResponse)
    {
        var rig=new Rig();
        rig.Door.SetConnection(noResponse);
        rig.Door.NoResponse=noResponse;
        rig.Door.ResponseTimeout=TimeSpan.FromMilliseconds(50);
        var code=await rig.OtpSessionAsync();
        await rig.Engine.TestOtpAsync(code.Code).WaitAsync(TimeSpan.FromSeconds(2));
        Assert.Equal(SessionState.Denied,rig.Engine.Snapshot!.State);
        Assert.NotNull(rig.Store.Otps.Single().ConsumedUtc);
        Assert.Equal(0,rig.Door.Snapshot.OpenCount);
        await Task.Delay(100);
        Assert.Equal(0,rig.Door.Snapshot.OpenCount);
    }
    [Fact]
    public async Task CancelDuringDoorDelayPreventsLateOpen()
    {
        var rig=new Rig();
        rig.Door.Delay=TimeSpan.FromSeconds(1);
        var code=await rig.OtpSessionAsync();
        var work=rig.Engine.TestOtpAsync(code.Code);
        await Task.Delay(40);
        await rig.Engine.CancelAsync();
        await work;
        Assert.Equal(0,rig.Door.Snapshot.OpenCount);
        Assert.NotNull(rig.Store.Otps.Single().ConsumedUtc);
        Assert.Equal("CancelledOrTimedOut",rig.Store.Commands.Single().Status);
    }
    [Fact]
    public async Task MockValidIdentityRequiresEnrollment()
    {
        var rig=new Rig();
        await rig.EnrollAsync();
        await rig.Engine.StartAsync();
        await rig.Engine.FaceAsync("input");
        Assert.Equal(SessionState.Granted,rig.Engine.Snapshot!.State);
        Assert.Equal(1,rig.Door.Snapshot.OpenCount);
    }
}
