using SmartDoor.Core;
using SmartDoor.Infrastructure;
using Xunit;

namespace SmartDoor.Tests;

public sealed class LiveAiFactAttribute:FactAttribute
{
    public LiveAiFactAttribute()
    {
        if(string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("PBL4_TEST_AI_URL"))) Skip="Chưa cấu hình PBL4_TEST_AI_URL; không chạy model thật.";
    }
}
public sealed class LiveAiTests
{
    [LiveAiFact]
    public async Task ExistingPythonModelsHealthAndInvalidInputs()
    {
        using var http=new HttpClient { BaseAddress=new Uri(Environment.GetEnvironmentVariable("PBL4_TEST_AI_URL")!),Timeout=TimeSpan.FromSeconds(60) };
        var ai=new RealAiClient(http);
        Assert.Contains("ready",await ai.HealthAsync(default));
        Assert.Empty(await ai.UsersAsync(default)); // Requires isolated test AI database.
        var image=Environment.GetEnvironmentVariable("PBL4_TEST_BLANK_IMAGE")!;
        var audio=Environment.GetEnvironmentVariable("PBL4_TEST_SILENT_AUDIO")!;
        var session=Guid.NewGuid();
        var error=await Assert.ThrowsAsync<AiInputException>(()=>ai.RecognizeAsync(image,session,0.363,0.05,default));
        Assert.Contains("không có mặt",error.Message);
        var voice=await ai.TranscribeAsync(audio,session,default);
        Assert.Equal(session.ToString(),voice.SessionId);
        Assert.Equal("retry",voice.Status);
        Assert.Null(voice.Digits);
    }
}
