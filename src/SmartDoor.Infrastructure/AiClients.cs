using System.Globalization;
using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using SmartDoor.Core;

namespace SmartDoor.Infrastructure;

public sealed class RealAiClient(HttpClient http):IAiClient
{
    public AiMode Mode=>AiMode.Real;
    public async Task<string> HealthAsync(CancellationToken ct)
    {
        using var response=await http.GetAsync("/health",ct);
        using var json=await ReadAsync(response,ct);
        return $"{json.RootElement.GetProperty("status").GetString()} · chống giả mạo chưa có";
    }
    public async Task<IReadOnlyList<AiUser>> UsersAsync(CancellationToken ct)
    {
        using var response=await http.GetAsync("/users",ct);
        using var json=await ReadAsync(response,ct);
        return json.RootElement.EnumerateArray().Select(x=>new AiUser(x.GetProperty("slot").GetInt32(),x.GetProperty("name").GetString()!,x.GetProperty("samples").GetInt32())).ToArray();
    }
    public async Task DeleteAsync(int slot,CancellationToken ct)
    {
        using var response=await http.DeleteAsync($"/users/{slot}",ct);
        using var json=await ReadAsync(response,ct);
        var root=json.RootElement;
        if(root.GetProperty("slot").GetInt32()!=slot||root.GetProperty("status").GetString() is not ("deleted" or "not_found")) throw new AiInputException("AI chưa xác nhận xóa mẫu.");
    }
    public async Task<EnrollmentResult> EnrollAsync(int slot,string name,IReadOnlyList<string> images,bool replace,Guid session,CancellationToken ct)
    {
        using var content=Form(session);
        content.Add(new StringContent(slot.ToString(CultureInfo.InvariantCulture)),"slot");
        content.Add(new StringContent(name),"name");
        content.Add(new StringContent(replace?"true":"false"),"replace");
        foreach(var image in images) AddFile(content,"images",image,10*1024*1024);
        using var response=await PostAsync("/face/enroll",content,ct);
        using var json=await ReadAsync(response,ct);
        var root=json.RootElement;
        return new(root.GetProperty("session_id").GetString()!,root.GetProperty("slot").GetInt32(),root.GetProperty("samples").GetInt32(),root.GetProperty("status").GetString()!);
    }
    public async Task<FaceResult> RecognizeAsync(string file,Guid session,double threshold,double margin,CancellationToken ct)
    {
        using var content=Form(session);
        AddFile(content,"image",file,10*1024*1024);
        content.Add(new StringContent(threshold.ToString(CultureInfo.InvariantCulture)),"threshold");
        content.Add(new StringContent(margin.ToString(CultureInfo.InvariantCulture)),"margin");
        using var response=await PostAsync("/face/recognize",content,ct);
        using var json=await ReadAsync(response,ct);
        var r=json.RootElement;
        var candidate=r.GetProperty("candidate");
        return new(r.GetProperty("session_id").GetString()!,r.GetProperty("request_id").GetString()!,r.GetProperty("elapsed_ms").GetInt64(),r.GetProperty("status").GetString()!,candidate.ValueKind==JsonValueKind.Null?null:candidate.GetProperty("slot").GetInt32(),r.GetProperty("liveness").GetString()!,candidate.ValueKind==JsonValueKind.Null?null:candidate.GetProperty("similarity").GetDouble());
    }
    public async Task<VoiceResult> TranscribeAsync(string file,Guid session,CancellationToken ct)
    {
        using var content=Form(session);
        AddFile(content,"audio",file,20*1024*1024);
        using var response=await PostAsync("/voice/transcribe",content,ct);
        using var json=await ReadAsync(response,ct);
        var r=json.RootElement;
        // Deliberately do not retain transcript or otp_verified/unlock_allowed as authority.
        return new(r.GetProperty("session_id").GetString()!,r.GetProperty("request_id").GetString()!,r.GetProperty("elapsed_ms").GetInt64(),r.GetProperty("status").GetString()!,r.GetProperty("digits").GetString());
    }
    private static MultipartFormDataContent Form(Guid session)
    {
        var form=new MultipartFormDataContent();
        form.Add(new StringContent(session.ToString()),"session_id");
        return form;
    }
    private static void AddFile(MultipartFormDataContent form,string field,string path,long limit)
    {
        var info=new FileInfo(path);
        if(!info.Exists||info.Length==0||info.Length>limit) throw new AiInputException("File không tồn tại, rỗng hoặc vượt giới hạn dung lượng.");
        var stream=new StreamContent(File.OpenRead(path));
        form.Add(stream,field,Path.GetFileName(path));
    }
    private async Task<HttpResponseMessage> PostAsync(string path,HttpContent content,CancellationToken ct)
    {
        try { return await http.PostAsync(path,content,ct); }
        catch(OperationCanceledException) when(!ct.IsCancellationRequested) { throw new TimeoutException(); }
    }
    private static async Task<JsonDocument> ReadAsync(HttpResponseMessage response,CancellationToken ct)
    {
        if(response.StatusCode==HttpStatusCode.UnprocessableEntity)
        {
            // Whitelist fixed input reasons; never echo arbitrary server bodies into logs.
            var body=await response.Content.ReadAsStringAsync(ct);
            var reason=body.Contains("Exactly one face required",StringComparison.Ordinal)?"Ảnh không có mặt hoặc có nhiều mặt (API yêu cầu đúng một mặt)":body.Contains("Invalid image",StringComparison.Ordinal)?"Ảnh hỏng/không giải mã được":body.Contains("Face too small",StringComparison.Ordinal)?"Khuôn mặt quá nhỏ":"Dữ liệu đầu vào bị AI từ chối (HTTP 422)";
            throw new AiInputException(reason);
        }
        response.EnsureSuccessStatusCode();
        return JsonDocument.Parse(await response.Content.ReadAsStringAsync(ct));
    }
}
public enum MockScenario { ValidIdentity, Unknown, NoFace, MultipleFaces, InvalidImage, IncompleteOtp, Unavailable, EnrollmentFailure, Delayed, DeleteFailure }
public sealed class MockAiClient:IAiClient
{
    private readonly Dictionary<int,AiUser> users=new();
    public AiMode Mode=>AiMode.Mock;
    public MockScenario Scenario { get; set; }=MockScenario.ValidIdentity;
    public int CandidateSlot { get; set; }=1;
    public string TestDigits { get; set; }="";
    public TimeSpan Delay { get; set; }=TimeSpan.FromSeconds(5);
    public bool IgnoreCancellation { get; set; }
    public Task<string> HealthAsync(CancellationToken ct)=>Scenario==MockScenario.Unavailable?Task.FromException<string>(new HttpRequestException()):Task.FromResult("Mock AI · dữ liệu kịch bản; không đo độ chính xác model");
    public Task<IReadOnlyList<AiUser>> UsersAsync(CancellationToken ct)
    {
        lock(users) return Task.FromResult<IReadOnlyList<AiUser>>(users.Values.OrderBy(x=>x.Slot).ToArray());
    }
    public Task<EnrollmentResult> EnrollAsync(int slot,string name,IReadOnlyList<string> images,bool replace,Guid session,CancellationToken ct)
    {
        if(Scenario==MockScenario.EnrollmentFailure) throw new AiInputException("Mô phỏng đăng ký thất bại");
        lock(users)
        {
            if(users.ContainsKey(slot)&&!replace) throw new AiInputException("Vị trí đã có mẫu; cần xác nhận thay mẫu.");
            users[slot]=new(slot,name,images.Count);
        }
        return Task.FromResult(new EnrollmentResult(session.ToString(),slot,images.Count,"enrolled"));
    }
    public Task DeleteAsync(int slot,CancellationToken ct)
    {
        if(Scenario==MockScenario.DeleteFailure) throw new AiInputException("Mô phỏng xóa mẫu thất bại");
        lock(users) users.Remove(slot);
        return Task.CompletedTask;
    }
    public async Task<FaceResult> RecognizeAsync(string file,Guid session,double threshold,double margin,CancellationToken ct)
    {
        var scenario=Scenario;
        if(scenario==MockScenario.Delayed) await Task.Delay(Delay,IgnoreCancellation?CancellationToken.None:ct);
        if(scenario==MockScenario.Unavailable) throw new HttpRequestException();
        if(scenario is MockScenario.NoFace or MockScenario.MultipleFaces or MockScenario.InvalidImage) throw new AiInputException(scenario==MockScenario.NoFace?"Mô phỏng: ảnh không có mặt":scenario==MockScenario.MultipleFaces?"Mô phỏng: ảnh có nhiều mặt":"Mô phỏng: ảnh hỏng");
        return new(session.ToString(),Guid.NewGuid().ToString(),0,scenario==MockScenario.Unknown?"unknown":"candidate",scenario==MockScenario.Unknown?null:CandidateSlot,"passed",scenario==MockScenario.Unknown?null:0.9);
    }
    public Task<VoiceResult> TranscribeAsync(string file,Guid session,CancellationToken ct)
    {
        if(Scenario==MockScenario.Unavailable) throw new HttpRequestException();
        return Task.FromResult(new VoiceResult(session.ToString(),Guid.NewGuid().ToString(),0,Scenario==MockScenario.IncompleteOtp?"retry":"decoded",Scenario==MockScenario.IncompleteOtp?null:TestDigits));
    }
}
