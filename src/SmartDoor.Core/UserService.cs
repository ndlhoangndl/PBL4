namespace SmartDoor.Core;

public sealed class UserService(IStore store,IAiClient ai,AuthorizationGate authorization)
{
    private readonly SemaphoreSlim mutations=authorization.Semaphore;
    public async Task SaveAsync(int slot,string name,bool enabled,CancellationToken ct=default)
    {
        if(slot is <1 or >4||string.IsNullOrWhiteSpace(name)||name.Trim().Length>100) throw new ArgumentException("Vị trí 1–4, tên 1–100 ký tự.");
        await mutations.WaitAsync(ct);
        try
        {
            var user=(await store.UsersAsync(ct)).SingleOrDefault(x=>x.Slot==slot)??new User { Slot=slot };
            // Python has no rename endpoint. Renaming requires re-enrollment to synchronize.
            if(user.Name!=name.Trim()) user.Synchronized=false;
            user.Name=name.Trim();
            user.Enabled=enabled;
            user.Revision++;
            await store.SaveUserAsync(user,ct);
        }
        finally { mutations.Release(); }
    }
    public async Task EnrollAsync(int slot,IReadOnlyList<string> images,bool replace,CancellationToken ct=default)
    {
        if(images.Count is <3 or >30) throw new ArgumentException("Chọn 3–30 ảnh cùng một người.");
        await mutations.WaitAsync(ct);
        try
        {
            var user=(await store.UsersAsync(ct)).Single(x=>x.Slot==slot);
            user.Synchronized=false;
            user.Revision++;
            await store.SaveUserAsync(user,ct); // Fail closed BEFORE Python mutation.
            var session=Guid.NewGuid();
            var result=await ai.EnrollAsync(slot,user.Name,images,replace,session,ct);
            var remote=(await ai.UsersAsync(ct)).SingleOrDefault(x=>x.Slot==slot);
            if(result.SessionId!=session.ToString()||result.Status!="enrolled"||result.Slot!=slot||result.Samples!=images.Count||remote?.Name!=user.Name||remote.Samples!=result.Samples)
                throw new InvalidOperationException("Đăng ký chưa đồng bộ; cần đăng ký lại.");
            user.Samples=result.Samples;
            user.TemplateMode=ai.Mode;
            user.Synchronized=true;
            user.Revision++;
            await store.SaveUserAsync(user,ct);
        }
        finally { mutations.Release(); }
    }
    public async Task DeleteAsync(int slot,CancellationToken ct=default)
    {
        await mutations.WaitAsync(ct);
        try
        {
            var user=(await store.UsersAsync(ct)).Single(x=>x.Slot==slot);
            user.Synchronized=false;
            user.Enabled=false;
            user.Revision++;
            await store.SaveUserAsync(user,ct);
            if(user.Samples>0&&user.TemplateMode!=ai.Mode) throw new InvalidOperationException("Đã chặn quyền; chọn đúng chế độ AI của bộ mẫu rồi xóa lại để tránh để sót mẫu ở backend khác.");
            await ai.DeleteAsync(slot,ct);
            if((await ai.UsersAsync(ct)).Any(x=>x.Slot==slot)) throw new InvalidOperationException("AI chưa xóa mẫu; người dùng bị vô hiệu hóa và chưa đồng bộ.");
            await store.DeleteUserAsync(slot,ct);
        }
        finally { mutations.Release(); }
    }
    public async Task ReconcileAsync(CancellationToken ct=default)
    {
        await mutations.WaitAsync(ct);
        try
        {
            var remote=await ai.UsersAsync(ct);
            foreach(var user in await store.UsersAsync(ct))
            {
                var match=remote.SingleOrDefault(x=>x.Slot==user.Slot);
                if(user.Synchronized&&(user.TemplateMode!=ai.Mode||match?.Name!=user.Name||match.Samples!=user.Samples))
                {
                    user.Synchronized=false;
                    user.Revision++;
                    await store.SaveUserAsync(user,ct);
                }
            }
        }
        finally { mutations.Release(); }
    }
}
