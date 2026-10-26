// Add to your .NET desktop project. Service must run on localhost:8000.
using System;
using System.IO;
using System.Net.Http;
using System.Threading.Tasks;

public class AiClient
{
    private readonly HttpClient client=new HttpClient
    {
        BaseAddress=new Uri("http://127.0.0.1:8000"),
        Timeout=TimeSpan.FromSeconds(60)
    };

    public async Task<string> RecognizeAsync(string imagePath,string sessionId)
    {
        using var content=new MultipartFormDataContent();
        using var stream=File.OpenRead(imagePath);
        content.Add(new StreamContent(stream),"image",Path.GetFileName(imagePath));
        content.Add(new StringContent("0.363"),"threshold");
        content.Add(new StringContent("0.05"),"margin");
        content.Add(new StringContent(sessionId),"session_id");
        using var response=await client.PostAsync("/face/recognize",content);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadAsStringAsync();
    }

    public async Task<string> TranscribeAsync(string audioPath,string sessionId)
    {
        using var content=new MultipartFormDataContent();
        using var stream=File.OpenRead(audioPath);
        content.Add(new StreamContent(stream),"audio",Path.GetFileName(audioPath));
        content.Add(new StringContent(sessionId),"session_id");
        using var response=await client.PostAsync("/voice/transcribe",content);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadAsStringAsync();
    }
}
