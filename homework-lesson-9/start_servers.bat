@echo off
setlocal

cd /d "%~dp0"

start "SearchMCP" cmd /k python mcp_servers/search_mcp.py
start "ReportMCP" cmd /k python mcp_servers/report_mcp.py
start "ACP Server" cmd /k python acp_server.py

echo Started SearchMCP, ReportMCP, and ACP Server in separate windows.
