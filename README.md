# Agents

## Requirements

### Install Python Requirements

`pip install -e requirements.txt`

### Install Browsers for Playwright

`playwright install`

## MCP Servers

You'll need to be in the root folder of the appropriate MCP server folder.

Run any of the MCP servers using the command:

`FASTMCP_HOST="<host-ip-address>" FASTMCP_PORT="<port>" python main.py`

This sets up an MCP server listening on the host address at the configured port.

## Invoke Agents

You'll need to be within the appropriate project root folder. Invoke the agents with the run_query.py script.

`python run_query.py --query "Some tesxt query you may have."`

If there are other params needed, the script will complain.
