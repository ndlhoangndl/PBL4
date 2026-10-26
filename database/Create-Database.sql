-- Run with sqlcmd/SSMS. Application also creates this empty schema via EF EnsureCreated.
-- Requires CREATE DATABASE permission. No users, secrets or biometric data are seeded.
IF DB_ID(N'Pbl4SmartDoor') IS NULL
    CREATE DATABASE [Pbl4SmartDoor];
GO
USE [Pbl4SmartDoor];
GO
:r schema.sql
