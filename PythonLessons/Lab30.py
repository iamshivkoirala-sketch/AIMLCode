import os, time, hashlib, warnings
import win32serviceutil, win32service, win32evtlog, win32evtlogutil

#warnings.filterwarnings("ignore", category=DeprecationWarning)
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
@tool
def restart_sql_server():
    """Starts or restarts the SQL Server (MSSQL$SQLEXPRESS) service."""
    service_name = "MSSQL$SQLEXPRESS"
    try:
        # Check status and restart
        status = win32serviceutil.QueryServiceStatus(service_name)[1]
        if status != win32service.SERVICE_STOPPED:
            win32serviceutil.RestartService(service_name)
        else:
            win32serviceutil.StartService(service_name)
            return "SQL Server service restart/start command issued successfully."
    except Exception as e:
        return f"Failed to restart SQL Server: {str(e)}"

@tool
def get_current_event_viewer_message(log_name: str = "Application"):
    """Gets the most recent ERROR entry from the Windows Event Viewer Application log."""
    server = 'localhost'
    # Open the log
    hand = win32evtlog.OpenEventLog(server, log_name)
    flags = win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ
    
    try:
        while True:
            events = win32evtlog.ReadEventLog(hand, flags, 0)
            if not events: break
            for event in events:
                # 1 = Error, 2 = Warning, 4 = Information
                if event.EventType == win32evtlog.EVENTLOG_ERROR_TYPE:
                    msg = win32evtlogutil.SafeFormatMessage(event, log_name)
                    return (f"Source: {event.SourceName}\n"
                            f"EventId: {event.EventID}\n"
                            f"Time: {event.TimeGenerated}\n"
                            f"Message: {msg}")
    finally:
        win32evtlog.CloseEventLog(hand)
    return "No error found in Application log."

tools = [restart_sql_server, get_current_event_viewer_message]
model = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("AIKEY"))
agent = create_react_agent(model, tools, prompt="""You are an on-call agent. 
    1 line headline, 2-4 bullets. If 'SQL Server stopped', restart it.""")

def monitor():
    print("Agent active. Monitoring Windows Event Log...")
    last_hash = None
    while True:
        raw = get_current_event_viewer_message.run("Application")
        if "No error found" in raw:
            time.sleep(30)
            continue
        cur_hash = hashlib.md5(raw.encode()).hexdigest()
        
        if cur_hash != last_hash:
            last_hash = cur_hash
            input_msg = {"messages": [("user", f"New error: {raw}")]}
            for s in agent.stream(input_msg):
                if "agent" in s:
                    print(f"\n[{time.strftime('%H:%M:%S')}] ALERT:\n{s['agent']['messages'][-1].content}")
        
        time.sleep(30)

monitor()