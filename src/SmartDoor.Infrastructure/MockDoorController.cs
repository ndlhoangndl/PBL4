using SmartDoor.Core;

namespace SmartDoor.Infrastructure;

public sealed class MockDoorController:IDoorController
{
    private readonly object gate=new();
    private readonly Dictionary<Guid,DoorReply> replies=new();
    private bool connected=true,person,open;
    private int opens;
    public bool NoResponse { get; set; }
    public TimeSpan Delay { get; set; }=TimeSpan.FromMilliseconds(250);
    public TimeSpan ResponseTimeout { get; set; }=TimeSpan.FromSeconds(3);
    public DoorSnapshot Snapshot
    {
        get { lock(gate) return new(connected,person,open,"Chưa xác định (không có cảm biến chốt)",opens); }
    }
    public void SetConnection(bool value) { lock(gate) connected=value; }
    public void SetPerson(bool value) { lock(gate) person=value; }
    public void Close() { lock(gate) open=false; }
    public async Task<DoorReply> SendOpenAsync(Guid commandId,Func<Func<DoorReply>,DoorReply> commitGuard,CancellationToken ct)
    {
        lock(gate)
        {
            if(replies.TryGetValue(commandId,out var existing)) return existing;
        }
        using var timeout=CancellationTokenSource.CreateLinkedTokenSource(ct);
        timeout.CancelAfter(ResponseTimeout);
        try
        {
            await Task.Delay(NoResponse?Timeout.InfiniteTimeSpan:Delay,timeout.Token);
        }
        catch(OperationCanceledException) when(!ct.IsCancellationRequested)
        {
            lock(gate)
            {
                if(replies.TryGetValue(commandId,out var existing)) return existing;
                return replies[commandId]=new("NoResponse",false);
            }
        }
        DoorReply reply=new("Cancelled",false);
        return commitGuard(()=>
        {
            lock(gate)
            {
                ct.ThrowIfCancellationRequested();
                if(replies.TryGetValue(commandId,out var existing)) return existing;
                if(!connected) reply=new("Disconnected",false);
                else
                {
                    open=true;
                    opens++;
                    reply=new("Acknowledged",true);
                }
                replies[commandId]=reply;
                return reply;
            }
        });
    }
}
