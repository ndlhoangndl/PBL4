using SmartDoor.Core;
using SmartDoor.Infrastructure;

namespace SmartDoor.Desktop;

public sealed class MainForm:Form
{
    private readonly AppController app;
    private readonly Label status=new() { AutoSize=false,Dock=DockStyle.Fill,Padding=new Padding(16),ForeColor=Color.FromArgb(25,70,78) };
    private readonly Label session=new() { AutoSize=false,Dock=DockStyle.Fill,Padding=new Padding(16) };
    private readonly Label feedback=new() { Dock=DockStyle.Bottom,Height=58,Padding=new Padding(16),BackColor=Color.FromArgb(228,241,237),Text="Chỉ nhận file thiết bị ngoài · mọi lệnh cửa là mô phỏng" };
    private readonly DataGridView userGrid=Grid();
    private readonly DataGridView otpGrid=Grid();
    private readonly DataGridView historyGrid=Grid();
    private readonly NumericUpDown slot=new() { Minimum=1,Maximum=4,Value=1,Width=60 };
    private readonly TextBox name=new() { Width=220,PlaceholderText="Tên người dùng" };
    private readonly CheckBox enabled=new() { Text="Cho phép truy cập",Checked=true,AutoSize=true };
    private readonly NumericUpDown minutes=new() { Minimum=1,Maximum=1440,Value=5,Width=80 };
    private readonly TextBox issuedCode=new() { Width=150,ReadOnly=true,PlaceholderText="Chỉ hiện khi vừa tạo" };
    private readonly TextBox testCode=new() { Width=150,MaxLength=6,PlaceholderText="OTP kiểm thử",UseSystemPasswordChar=true };
    private readonly ComboBox methods=Choices("Tất cả","Face","VoiceOtp","TestOtp","Exit","Session");
    private readonly ComboBox results=Choices("Tất cả","Granted","Denied","Retry","Failed","Discarded","Cancelled","TimedOut");
    private readonly DateTimePicker from=new() { Width=180,Format=DateTimePickerFormat.Custom,CustomFormat="dd/MM/yyyy HH:mm",Value=DateTime.Today.AddDays(-7) };
    private readonly DateTimePicker to=new() { Width=180,Format=DateTimePickerFormat.Custom,CustomFormat="dd/MM/yyyy HH:mm",Value=DateTime.Now.AddDays(1) };
    private readonly System.Windows.Forms.Timer timer=new() { Interval=500 };
    private bool ticking,closing;
    private DateTime codeVisibleUntil;
    private readonly HashSet<Control> running=new();
    public MainForm(AppController controller)
    {
        app=controller;
        Text="PBL4 · Hệ thống khóa cửa thông minh";
        Font=new Font("Segoe UI",10);
        BackColor=Color.FromArgb(245,248,250);
        Size=new Size(1220,810);
        MinimumSize=new Size(1050,720);
        StartPosition=FormStartPosition.CenterScreen;
        var header=new Label { Dock=DockStyle.Top,Height=82,Text="PBL4  /  KHÓA CỬA THÔNG MINH\nQuản lý truy cập · đầu vào bằng file · cửa mô phỏng",Font=new Font("Segoe UI",14,FontStyle.Bold),BackColor=Color.FromArgb(20,71,78),ForeColor=Color.White,Padding=new Padding(22,10,0,0) };
        var tabs=new TabControl { Dock=DockStyle.Fill,Padding=new Point(22,10) };
        tabs.TabPages.Add(Overview());
        tabs.TabPages.Add(Users());
        tabs.TabPages.Add(Otps());
        tabs.TabPages.Add(History());
        tabs.TabPages.Add(Configuration());
        Controls.Add(tabs);
        Controls.Add(feedback);
        Controls.Add(header);
        userGrid.SelectionChanged+=(_,_)=>
        {
            if(userGrid.CurrentRow?.DataBoundItem is User user)
            {
                slot.Value=user.Slot;
                name.Text=user.Name;
                enabled.Checked=user.Enabled;
            }
        };
        tabs.SelectedIndexChanged+=async(_,_)=>
        {
            issuedCode.Clear();
            if(tabs.SelectedIndex==1) await RunAsync(userGrid,RefreshUsersAsync);
            if(tabs.SelectedIndex==2) await RunAsync(otpGrid,RefreshOtpsAsync);
        };
        Shown+=async(_,_)=>
        {
            await RunAsync(this,app.CheckAsync);
            timer.Start();
        };
        timer.Tick+=async(_,_)=>
        {
            if(issuedCode.Text.Length>0&&DateTime.UtcNow>=codeVisibleUntil) issuedCode.Clear();
            UpdateStatus();
            if(ticking) return;
            ticking=true;
            try { await app.Engine.TickAsync(); }
            catch(Exception) { feedback.Text="Không lưu được trạng thái phiên; kiểm tra SQL Server."; }
            finally { ticking=false; }
        };
        FormClosing+=async(_,e)=>
        {
            if(closing) return;
            e.Cancel=true;
            timer.Stop();
            try { await app.Engine.CancelAsync(); }
            catch(Exception) { }
            closing=true;
            Close();
        };
        UpdateStatus();
    }
    private static DataGridView Grid()=>new() { Dock=DockStyle.Fill,ReadOnly=true,AllowUserToAddRows=false,AllowUserToDeleteRows=false,AutoSizeColumnsMode=DataGridViewAutoSizeColumnsMode.DisplayedCells,SelectionMode=DataGridViewSelectionMode.FullRowSelect,MultiSelect=false,BackgroundColor=Color.White,BorderStyle=BorderStyle.None,RowHeadersVisible=false };
    private static ComboBox Choices(params string[] choices)
    {
        var box=new ComboBox { Width=140,DropDownStyle=ComboBoxStyle.DropDownList };
        box.Items.AddRange(choices);
        box.SelectedIndex=0;
        return box;
    }
    private static FlowLayoutPanel Bar(params Control[] controls)
    {
        var bar=new FlowLayoutPanel { Dock=DockStyle.Top,AutoSize=true,Padding=new Padding(12),WrapContents=true,BackColor=Color.White };
        bar.Controls.AddRange(controls);
        return bar;
    }
    private static Label Note(string text)=>new() { Text=text,AutoSize=true,MaximumSize=new Size(1000,0),Margin=new Padding(5,10,5,10),ForeColor=Color.FromArgb(58,78,91) };
    private Button ActionButton(string text,Func<Task> action)
    {
        var button=new Button { Text=text,AutoSize=true,MinimumSize=new Size(100,36),FlatStyle=FlatStyle.Flat,BackColor=Color.FromArgb(231,242,241),ForeColor=Color.FromArgb(20,71,78),Margin=new Padding(4) };
        button.Click+=async(_,_)=>await RunAsync(button,action);
        return button;
    }
    private static TabPage Page(string text)=>new(text) { BackColor=Color.FromArgb(245,248,250),Padding=new Padding(10) };
    private TabPage Overview()
    {
        var page=Page("Tổng quan");
        var cards=new TableLayoutPanel { Dock=DockStyle.Fill,ColumnCount=1,RowCount=3 };
        cards.RowStyles.Add(new RowStyle(SizeType.Absolute,150));
        cards.RowStyles.Add(new RowStyle(SizeType.Absolute,140));
        cards.RowStyles.Add(new RowStyle(SizeType.Percent,100));
        cards.Controls.Add(status,0,0);
        cards.Controls.Add(session,0,1);
        cards.Controls.Add(Note("Ảnh từ camera ngoài và bản ghi từ module ngoài. AI thật chỉ trả ứng viên khuôn mặt vì chưa có chống giả mạo. OTP xác thực nội dung mã, không xác minh người nói. Ngưỡng mặt là cấu hình thử nghiệm."),0,2);
        page.Controls.Add(cards);
        page.Controls.Add(Bar(ActionButton("Bắt đầu phiên",app.StartAsync),ActionButton("Hủy phiên",()=>app.Engine.CancelAsync()),ActionButton("Chuyển sang OTP",app.Engine.UseOtpAsync),ActionButton("Chọn ảnh…",async()=>
        {
            var files=PickFiles(false,false);
            if(files.Length>0) await app.FaceAsync(files[0]);
        }),ActionButton("Chọn âm thanh…",async()=>
        {
            var files=PickFiles(true,false);
            if(files.Length>0) await app.AudioAsync(files[0]);
        }),ActionButton("Kiểm tra dịch vụ",app.CheckAsync)));
        return page;
    }
    private TabPage Users()
    {
        var page=Page("Người dùng");
        page.Controls.Add(userGrid);
        page.Controls.Add(Bar(ActionButton("Lưu tên / quyền",async()=>
        {
            await app.SaveUserAsync((int)slot.Value,name.Text,enabled.Checked);
            await RefreshUsersAsync();
        }),ActionButton("Đăng ký 3–30 ảnh",async()=>await EnrollAsync(false)),ActionButton("Thay bộ mẫu…",async()=>await EnrollAsync(true)),ActionButton("Xóa người dùng…",async()=>
        {
            if(MessageBox.Show($"Xóa danh tính và mẫu AI vị trí {slot.Value}?","Xác nhận",MessageBoxButtons.YesNo,MessageBoxIcon.Warning)!=DialogResult.Yes) return;
            await app.DeleteUserAsync((int)slot.Value);
            await RefreshUsersAsync();
        }),ActionButton("Tải lại",RefreshUsersAsync)));
        page.Controls.Add(Bar(Note("Vị trí"),slot,name,enabled));
        page.Controls.Add(Bar(Note("Tối đa 4 người. Đổi tên yêu cầu đăng ký/thay mẫu lại để đồng bộ với Python. Mẫu Mock chỉ là metadata kiểm thử, không phải embedding. Cột Synchronized=false chặn xác thực mặt.")));
        return page;
    }
    private async Task EnrollAsync(bool replace)
    {
        if(replace&&MessageBox.Show("Thay toàn bộ mẫu của vị trí đang chọn? Xác nhận mọi ảnh thuộc cùng người.","Thay mẫu",MessageBoxButtons.YesNo,MessageBoxIcon.Warning)!=DialogResult.Yes) return;
        var files=PickFiles(false,true);
        if(files.Length==0) return;
        if(MessageBox.Show($"Đăng ký {files.Length} ảnh cho vị trí {slot.Value}. Các ảnh đều thuộc cùng người?","Xác nhận danh tính",MessageBoxButtons.YesNo)!=DialogResult.Yes) return;
        try { await app.EnrollAsync((int)slot.Value,files,replace); }
        finally { await RefreshUsersAsync(); }
    }
    private TabPage Otps()
    {
        var page=Page("OTP");
        page.Controls.Add(otpGrid);
        page.Controls.Add(Bar(Note("Thời hạn (phút)"),minutes,ActionButton("Sinh OTP",async()=>
        {
            var created=await app.CreateOtpAsync((int)minutes.Value);
            issuedCode.Text=created.Code;
            codeVisibleUntil=DateTime.UtcNow.AddSeconds(60);
            await RefreshOtpsAsync();
            feedback.Text="OTP mới hiện trong 60 giây. Lịch sử chỉ lưu ID, thời hạn và trạng thái.";
        }),issuedCode,ActionButton("Sao chép mã mới",()=>
        {
            if(issuedCode.Text.Length==6) Clipboard.SetText(issuedCode.Text);
            return Task.CompletedTask;
        }),ActionButton("Thu hồi mã chọn",async()=>
        {
            if(otpGrid.CurrentRow?.Cells["Id"].Value is Guid id) await app.RevokeOtpAsync(id);
            issuedCode.Clear();
            await RefreshOtpsAsync();
        }),ActionButton("Tải lại",RefreshOtpsAsync)));
        page.Controls.Add(Bar(Note("Mã 6 chữ số, dùng một lần; giới hạn 3 lần thử sai/phiên. Trạng thái Active, Consumed, Expired, Revoked. Mã rõ không lưu trong SQL hoặc log.")));
        return page;
    }
    private TabPage History()
    {
        var page=Page("Lịch sử");
        page.Controls.Add(historyGrid);
        page.Controls.Add(Bar(Note("Từ"),from,Note("Đến"),to,methods,results,ActionButton("Lọc lịch sử",async()=>
        {
            if(from.Value>to.Value) throw new ArgumentException("Khoảng thời gian không hợp lệ.");
            historyGrid.DataSource=await app.HistoryAsync(from.Value,to.Value,methods.Text,results.Text);
        })));
        page.Controls.Add(Bar(Note("Thời gian hiển thị theo giờ máy; lưu UTC. Nguồn AI và trạng thái lệnh cửa tách riêng. Mọi lệnh thiết bị trong giai đoạn này là mô phỏng. Tối đa 1.000 sự kiện mỗi lần lọc.")));
        return page;
    }
    private TabPage Configuration()
    {
        var page=Page("Cấu hình và kiểm thử");
        var layout=new FlowLayoutPanel { Dock=DockStyle.Fill,FlowDirection=FlowDirection.TopDown,WrapContents=false,AutoScroll=true };
        var mode=Choices("Mock","Real");
        mode.SelectedItem=app.Settings.AiMode.ToString();
        var url=new TextBox { Width=340,Text=app.Settings.AiUrl };
        var connection=new TextBox { Width=680,Text=app.Settings.ConnectionString,UseSystemPasswordChar=true };
        var timeout=new NumericUpDown { Minimum=1,Maximum=300,Value=app.Settings.TimeoutSeconds,Width=70 };
        var duration=new NumericUpDown { Minimum=5,Maximum=900,Value=app.Settings.SessionSeconds,Width=70 };
        var threshold=new NumericUpDown { Minimum=0,Maximum=1,DecimalPlaces=3,Increment=0.001m,Value=(decimal)app.Settings.Threshold,Width=80 };
        var margin=new NumericUpDown { Minimum=0,Maximum=1,DecimalPlaces=3,Increment=0.001m,Value=(decimal)app.Settings.Margin,Width=80 };
        layout.Controls.Add(Bar(Note("Chế độ AI"),mode,Note("URL AI"),url));
        layout.Controls.Add(Bar(Note("SQL connection string"),connection));
        layout.Controls.Add(Bar(Note("Timeout AI (s)"),timeout,Note("Phiên (s)"),duration,Note("Cosine"),threshold,Note("Margin"),margin));
        layout.Controls.Add(Bar(ActionButton("Lưu cấu hình",async()=>
        {
            try { await app.Engine.CancelAsync(); }
            catch(Exception) { feedback.Text="Phiên đã bị hủy trong bộ nhớ nhưng không lưu được audit; vẫn cho sửa cấu hình database."; }
            var settings=new AppSettings { AiMode=Enum.Parse<AiMode>(mode.Text),AiUrl=url.Text.Trim(),ConnectionString=connection.Text.Trim(),TimeoutSeconds=(int)timeout.Value,SessionSeconds=(int)duration.Value,Threshold=(double)threshold.Value,Margin=(double)margin.Value };
            settings.Save();
            MessageBox.Show("Đã lưu. Đóng và mở lại ứng dụng để áp dụng cấu hình.","Cấu hình");
        }),ActionButton("Kiểm tra dịch vụ",app.CheckAsync)));
        layout.Controls.Add(Note("KỊCH BẢN MOCK AI · chỉ áp dụng khi chế độ hiện tại là Mock"));
        var scenarios=new FlowLayoutPanel { AutoSize=true,MaximumSize=new Size(1080,0) };
        foreach(var item in new (string Name,MockScenario Value)[] { ("Người hợp lệ",MockScenario.ValidIdentity),("Người lạ",MockScenario.Unknown),("Không có mặt",MockScenario.NoFace),("Nhiều mặt",MockScenario.MultipleFaces),("Ảnh lỗi",MockScenario.InvalidImage),("OTP thiếu số",MockScenario.IncompleteOtp),("AI không sẵn sàng",MockScenario.Unavailable),("Đăng ký lỗi",MockScenario.EnrollmentFailure),("AI đến muộn",MockScenario.Delayed) })
            scenarios.Controls.Add(ActionButton(item.Name,()=>
            {
                if(app.Mock==null) throw new InvalidOperationException("Chọn Mock và khởi động lại để chạy kịch bản.");
                app.Mock.Scenario=item.Value;
                app.Mock.CandidateSlot=(int)slot.Value;
                feedback.Text=$"Mock AI: {item.Name}; ứng viên vị trí {slot.Value}. Bắt đầu phiên rồi gửi file để thử.";
                return Task.CompletedTask;
            }));
        layout.Controls.Add(scenarios);
        layout.Controls.Add(Note("Người hợp lệ cần có người dùng đã đăng ký metadata Mock và được cấp quyền. Vị trí lấy từ tab Người dùng. Kịch bản không tự tạo danh tính."));
        layout.Controls.Add(Bar(Note("ĐẦU VÀO OTP KIỂM THỬ (không nhận diện giọng nói)"),testCode,ActionButton("Gửi OTP kiểm thử",async()=>
        {
            var input=testCode.Text;
            testCode.Clear();
            await app.TestOtpAsync(input);
        }),ActionButton("Dùng mã làm kết quả Mock voice",()=>
        {
            if(app.Mock==null) throw new InvalidOperationException("Cần chế độ Mock.");
            app.Mock.TestDigits=testCode.Text;
            testCode.Clear();
            return Task.CompletedTask;
        })));
        layout.Controls.Add(Bar(ActionButton("Có người trước cửa",()=> { app.Door.SetPerson(true); return Task.CompletedTask; }),ActionButton("Không có người",()=> { app.Door.SetPerson(false); return Task.CompletedTask; }),ActionButton("Đóng cửa mô phỏng",()=> { app.Door.Close(); return Task.CompletedTask; }),ActionButton("Nút EXIT",app.ExitAsync)));
        layout.Controls.Add(Bar(ActionButton("Cửa mất kết nối",()=> { app.Door.SetConnection(false); return Task.CompletedTask; }),ActionButton("Cửa kết nối lại",()=> { app.Door.SetConnection(true); app.Door.NoResponse=false; return Task.CompletedTask; }),ActionButton("Lệnh không phản hồi",()=> { app.Door.NoResponse=true; return Task.CompletedTask; }),ActionButton("DB không sẵn sàng",async()=> { app.SimulateDatabaseFailure=true; await app.Engine.CancelAsync(); await app.CheckAsync(); }),ActionButton("Khôi phục DB",async()=> { app.SimulateDatabaseFailure=false; await app.CheckAsync(); })));
        page.Controls.Add(layout);
        return page;
    }
    private static string[] PickFiles(bool audio,bool multiple)
    {
        using var dialog=new OpenFileDialog { Multiselect=multiple,Filter=audio?"Âm thanh ngoài|*.wav;*.mp3;*.flac;*.m4a;*.ogg|Tất cả|*.*":"Ảnh camera ngoài|*.jpg;*.jpeg;*.png;*.bmp;*.webp|Tất cả|*.*" };
        return dialog.ShowDialog()==DialogResult.OK?dialog.FileNames:[];
    }
    private async Task RefreshUsersAsync()
    {
        userGrid.DataSource=await app.UsersAsync();
        var labels=new Dictionary<string,string> { ["Slot"]="Vị trí",["Name"]="Tên",["Enabled"]="Quyền",["Synchronized"]="Đồng bộ",["Samples"]="Số mẫu",["TemplateMode"]="Nguồn mẫu",["Revision"]="Phiên bản" };
        foreach(DataGridViewColumn column in userGrid.Columns) if(labels.TryGetValue(column.Name,out var label)) column.HeaderText=label;
    }
    private async Task RefreshOtpsAsync()=>otpGrid.DataSource=await app.OtpRowsAsync();
    private async Task RunAsync(Control source,Func<Task> action)
    {
        if(!running.Add(source)) return;
        if(source is Button) source.Enabled=false;
        var previousFeedback=feedback.Text;
        try
        {
            await action();
            if(feedback.Text==previousFeedback) feedback.Text="Thao tác kết thúc · xem trạng thái phiên/kết quả · cửa mô phỏng.";
        }
        catch(Exception ex)
        {
            var message=ex is ArgumentException or InvalidOperationException or AiInputException?ex.Message:ex is OperationCanceledException?"Thao tác bị hủy hoặc hết thời gian":$"Thao tác thất bại ({ex.GetType().Name}). Kiểm tra SQL Server, dịch vụ AI và cấu hình; dữ liệu có thể chưa đồng bộ.";
            feedback.Text=message;
            MessageBox.Show(message,"PBL4 · kết quả thao tác",MessageBoxButtons.OK,MessageBoxIcon.Warning);
        }
        finally
        {
            running.Remove(source);
            if(!source.IsDisposed&&source is Button) source.Enabled=true;
            if(!IsDisposed) UpdateStatus();
        }
    }
    private void UpdateStatus()
    {
        var door=app.Door.Snapshot;
        status.Text=$"CHẾ ĐỘ: {(app.Settings.AiMode==AiMode.Mock?"MOCK AI — MÔ PHỎNG":"AI THẬT — chưa có chống giả mạo")}\nAI: {app.AiStatus}\nDatabase: {app.DatabaseStatus}\nCửa MÔ PHỎNG: {(door.Connected?"kết nối":"mất kết nối")} · {(door.Open?"mở":"đóng")} · Có người: {door.PersonPresent} · Số lượt mở: {door.OpenCount}\nChốt: {door.BoltPosition}";
        var current=app.Engine.Snapshot;
        session.Text=current==null?"PHIÊN XÁC THỰC: Idle":$"PHIÊN: {current.Id}\nTrạng thái: {current.State} · Thử OTP sai: {current.FailedAttempts}/3\nHết hạn: {current.ExpiresUtc.ToLocalTime():dd/MM/yyyy HH:mm:ss}\nKết quả: {current.Reason}";
    }
    protected override void Dispose(bool disposing)
    {
        if(disposing) timer.Dispose();
        base.Dispose(disposing);
    }
}
