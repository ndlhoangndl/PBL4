namespace SmartDoor.Desktop;

internal static class Program
{
    [STAThread]
    private static void Main()
    {
        ApplicationConfiguration.Initialize();
        using var singleInstance=new Mutex(true,"Local\\Pbl4SmartDoorDesktop",out var firstInstance);
        if(!firstInstance)
        {
            MessageBox.Show("Ứng dụng đã chạy. Chỉ một tiến trình quản lý cửa tại một thời điểm.","PBL4");
            return;
        }
        try
        {
            using var controller=new AppController(AppSettings.Load());
            Application.Run(new MainForm(controller));
        }
        catch(Exception ex)
        {
            MessageBox.Show($"Không khởi động được ứng dụng ({ex.GetType().Name}). Kiểm tra appsettings.json, URL AI và khóa HMAC/DPAPI. Xem README.","PBL4",MessageBoxButtons.OK,MessageBoxIcon.Error);
        }
    }
}
