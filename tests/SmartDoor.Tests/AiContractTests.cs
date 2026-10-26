using System.Net;
using System.Text;
using SmartDoor.Core;
using SmartDoor.Infrastructure;
using Xunit;

namespace SmartDoor.Tests;

public sealed class AiContractTests
{
    private sealed class Handler(Func<HttpRequestMessage,Task<HttpResponseMessage>> respond):HttpMessageHandler
    {
        protected override Task<HttpResponseMessage> SendAsync(HttpRequestMessage request,CancellationToken ct)=>respond(request);
    }
    private static HttpResponseMessage Json(string body)=>new(HttpStatusCode.OK) { Content=new StringContent(body,Encoding.UTF8,"application/json") };
    [Fact]
    public async Task EnrollmentUsesRepeatedImagesAndExpectedResponseSchema()
    {
        var session=Guid.NewGuid();
        using var http=new HttpClient(new Handler(async request=>
        {
            Assert.Equal("/face/enroll",request.RequestUri!.AbsolutePath);
            var parts=(MultipartFormDataContent)request.Content!;
            string Field(HttpContent part)=>part.Headers.ContentDisposition!.Name!.Trim('"');
            Assert.Equal(3,parts.Count(x=>Field(x)=="images"));
            Assert.Equal("1",await parts.Single(x=>Field(x)=="slot").ReadAsStringAsync());
            Assert.Equal("Test",await parts.Single(x=>Field(x)=="name").ReadAsStringAsync());
            Assert.Equal("true",await parts.Single(x=>Field(x)=="replace").ReadAsStringAsync());
            Assert.Equal(session.ToString(),await parts.Single(x=>Field(x)=="session_id").ReadAsStringAsync());
            return Json($$"""{"status":"enrolled","slot":1,"samples":3,"unlock_allowed":false,"session_id":"{{session}}","request_id":"r","elapsed_ms":7}""");
        })) { BaseAddress=new Uri("http://127.0.0.1:8000") };
        var file=Path.GetTempFileName();
        try
        {
            await File.WriteAllTextAsync(file,"multipart contract placeholder only");
            var response=await new RealAiClient(http).EnrollAsync(1,"Test",[file,file,file],true,session,default);
            Assert.Equal("enrolled",response.Status);
            Assert.Equal(3,response.Samples);
            Assert.Equal(session.ToString(),response.SessionId);
        }
        finally { File.Delete(file); }
    }
    [Fact]
    public async Task RealFaceCandidateCannotAuthorizeDoorEvenIfLivenessClaimsPassed()
    {
        var store=new MemoryStore();
        var id=Guid.Empty;
        using var faceHttp=new HttpClient(new Handler(_=>Task.FromResult(Json($$"""{"session_id":"{{id}}","request_id":"r","elapsed_ms":5,"status":"candidate","candidate":{"slot":1,"name":"Test","similarity":0.9},"liveness":"passed","unlock_allowed":true}""")))) { BaseAddress=new Uri("http://127.0.0.1:8000") };
        var ai=new RealAiClient(faceHttp);
        var door=new MockDoorController();
        var otp=new OtpService(store,System.Security.Cryptography.RandomNumberGenerator.GetBytes(32),TimeProvider.System);
        var engine=new AuthenticationEngine(store,ai,door,otp,TimeProvider.System,new AuthorizationGate());
        await engine.StartAsync();
        id=engine.Snapshot!.Id;
        var file=Path.GetTempFileName();
        try
        {
            await File.WriteAllTextAsync(file,"API contract test placeholder; no biometric image");
            await engine.FaceAsync(file);
            Assert.Equal(SessionState.AwaitingOtp,engine.Snapshot!.State);
            Assert.Equal(0,door.Snapshot.OpenCount);
        }
        finally { File.Delete(file); }
    }
    [Fact]
    public async Task MultipartUsesPythonFieldNamesAndVoiceDropsTranscript()
    {
        var id=Guid.NewGuid();
        using var http=new HttpClient(new Handler(async request=>
        {
            Assert.Equal("/voice/transcribe",request.RequestUri!.AbsolutePath);
            var parts=(MultipartFormDataContent)request.Content!;
            var names=parts.Select(x=>x.Headers.ContentDisposition!.Name!.Trim('"')).ToArray();
            Assert.Contains("audio",names);
            Assert.Contains("session_id",names);
            Assert.Equal(id.ToString(),await parts.Single(x=>x.Headers.ContentDisposition!.Name!.Trim('"')=="session_id").ReadAsStringAsync());
            return Json($$"""{"session_id":"{{id}}","request_id":"r","elapsed_ms":10,"status":"decoded","transcript":"sensitive code","digits":"012345","otp_verified":false,"unlock_allowed":false}""");
        })) { BaseAddress=new Uri("http://127.0.0.1:8000") };
        var client=new RealAiClient(http);
        var file=Path.GetTempFileName();
        try
        {
            await File.WriteAllTextAsync(file,"not real audio; contract only");
            var response=await client.TranscribeAsync(file,id,default);
            Assert.Equal("012345",response.Digits);
            Assert.DoesNotContain("sensitive code",System.Text.Json.JsonSerializer.Serialize(response));
        }
        finally { File.Delete(file); }
    }
}
