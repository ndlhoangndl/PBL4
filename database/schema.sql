CREATE TABLE [AuthenticationSessions] (
    [Id] uniqueidentifier NOT NULL,
    [StartedUtc] datetime2 NOT NULL,
    [ExpiresUtc] datetime2 NOT NULL,
    [State] int NOT NULL,
    [Mode] int NOT NULL,
    [FailedAttempts] int NOT NULL,
    [Reason] nvarchar(max) NOT NULL,
    CONSTRAINT [PK_AuthenticationSessions] PRIMARY KEY ([Id])
);
GO


CREATE TABLE [OtpCodes] (
    [Id] uniqueidentifier NOT NULL,
    [Digest] nchar(64) NOT NULL,
    [CreatedUtc] datetime2 NOT NULL,
    [ExpiresUtc] datetime2 NOT NULL,
    [ConsumedUtc] datetime2 NULL,
    [Revoked] bit NOT NULL,
    CONSTRAINT [PK_OtpCodes] PRIMARY KEY ([Id])
);
GO


CREATE TABLE [Users] (
    [Slot] int NOT NULL,
    [Name] nvarchar(100) NOT NULL,
    [Enabled] bit NOT NULL,
    [Synchronized] bit NOT NULL,
    [Samples] int NOT NULL,
    [TemplateMode] int NOT NULL,
    [Revision] bigint NOT NULL,
    CONSTRAINT [PK_Users] PRIMARY KEY ([Slot]),
    CONSTRAINT [CK_Users_Slot] CHECK ([Slot] BETWEEN 1 AND 4)
);
GO


CREATE TABLE [AccessEvents] (
    [Id] uniqueidentifier NOT NULL,
    [SessionId] uniqueidentifier NOT NULL,
    [TimestampUtc] datetime2 NOT NULL,
    [UserSlot] int NULL,
    [Method] nvarchar(30) NOT NULL,
    [Mode] int NOT NULL,
    [Result] nvarchar(30) NOT NULL,
    [Reason] nvarchar(max) NOT NULL,
    [CommandId] uniqueidentifier NULL,
    CONSTRAINT [PK_AccessEvents] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_AccessEvents_AuthenticationSessions_SessionId] FOREIGN KEY ([SessionId]) REFERENCES [AuthenticationSessions] ([Id]) ON DELETE NO ACTION
);
GO


CREATE TABLE [DoorCommands] (
    [Id] uniqueidentifier NOT NULL,
    [SessionId] uniqueidentifier NOT NULL,
    [CreatedUtc] datetime2 NOT NULL,
    [Status] nvarchar(50) NOT NULL,
    [Simulated] bit NOT NULL,
    CONSTRAINT [PK_DoorCommands] PRIMARY KEY ([Id]),
    CONSTRAINT [FK_DoorCommands_AuthenticationSessions_SessionId] FOREIGN KEY ([SessionId]) REFERENCES [AuthenticationSessions] ([Id]) ON DELETE NO ACTION
);
GO


CREATE INDEX [IX_AccessEvents_SessionId] ON [AccessEvents] ([SessionId]);
GO


CREATE INDEX [IX_AccessEvents_TimestampUtc] ON [AccessEvents] ([TimestampUtc]);
GO


CREATE INDEX [IX_DoorCommands_SessionId] ON [DoorCommands] ([SessionId]);
GO


CREATE UNIQUE INDEX [IX_OtpCodes_Digest] ON [OtpCodes] ([Digest]);
GO


