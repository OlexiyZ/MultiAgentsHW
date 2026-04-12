#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python mcp_servers/search_mcp.py &
SEARCH_PID=$!

python mcp_servers/report_mcp.py &
REPORT_PID=$!

python acp_server.py &
ACP_PID=$!

echo "Started SearchMCP pid=${SEARCH_PID}, ReportMCP pid=${REPORT_PID}, ACP Server pid=${ACP_PID}."
echo "Press Ctrl+C to stop all servers."

trap 'kill "${SEARCH_PID}" "${REPORT_PID}" "${ACP_PID}" 2>/dev/null || true' INT TERM EXIT
wait
