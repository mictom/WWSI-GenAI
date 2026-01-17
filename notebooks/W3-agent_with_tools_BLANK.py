import os
import uuid
import pandas as pd
import csv
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(temperature=0, model_name="gpt-4o", streaming=False)

# File path for our mock database
RESERVATIONS_FILE = "reservations.csv"

# Initialize the CSV file if it doesn't exist
def initialize_csv():
    if not os.path.exists(RESERVATIONS_FILE):
        with open(RESERVATIONS_FILE, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["reservation_id", "reservation_date", "planned_trip_date",
                             "trip_destination", "description"])

# Tool to save a reservation
@tool
def save_reservation(planned_trip_date: str, trip_destination: str, description: str) -> str:
    """
    Save a new trip reservation.

    Args:
        planned_trip_date: The date of the planned trip (YYYY-MM-DD format)
        trip_destination: Destination of the trip
        description: Additional details about the trip

    Returns:
        A confirmation message with the reservation ID
    """
    initialize_csv()

    # Generate a unique reservation ID
    reservation_id = str(uuid.uuid4())[:8]
    reservation_date = datetime.now().strftime('%Y-%m-%d')

    # Save to CSV
    with open(RESERVATIONS_FILE, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([reservation_id, reservation_date, planned_trip_date,
                         trip_destination, description])
    print(f"Saved reservation to CSV: {RESERVATIONS_FILE}")

    return f"Reservation created successfully! Your reservation ID is: {reservation_id}"

# Tool to read a reservation
@tool
def read_reservation(reservation_id: str) -> str:
    """
    Look up a reservation by ID.

    Args:
        reservation_id: The unique ID of the reservation to look up

    Returns:
        Details of the reservation or an error message if not found
    """
    initialize_csv()

    try:
        df = pd.read_csv(RESERVATIONS_FILE)
        reservation = df[df['reservation_id'] == reservation_id]

        if reservation.empty:
            return f"No reservation found with ID: {reservation_id}"

        # Get the first (and should be only) matching reservation
        res = reservation.iloc[0]
        return f"Reservation found:\nID: {res['reservation_id']}\nBooked on: {res['reservation_date']}\nTrip date: {res['planned_trip_date']}\nDestination: {res['trip_destination']}\nDetails: {res['description']}"

    except Exception as e:
        return f"Error looking up reservation: {str(e)}"


## TODO 1: Add the tools you want the agent to use to this list
## Hint: We defined two tools above - save_reservation and read_reservation
tools = []

# Create a tools dictionary for easy lookup
tools_dict = {tool.name: tool for tool in tools}

## TODO 2: Bind the tools to the LLM so it knows what tools are available
## Hint: Use llm.bind_tools() with the tools list
llm_with_tools = None  # Replace None with the correct binding

# System prompt for the agent
SYSTEM_PROMPT = """You are a helpful travel booking assistant.
You can help users make new travel reservations or look up existing ones.

To make a new reservation, you need:
1. The planned trip date (in YYYY-MM-DD format)
2. The destination
3. Any additional details or description

To look up an existing reservation, you need the reservation ID.

If you are not sure which tool is best for the task use multiple and then select best output.
When you are using a tool, remember to provide all relevant context for the tool to execute the task.

If you gave the user some recommendations in previous messages and he agrees with them use those recommendations in your actions.
When analyzing tool output, compare it with Human question, if it only partially answered it explain it to the user."""


def run_agent_with_query(query: str, verbose: bool = True) -> dict:
    """
    Run the agent with a query using a simple tool-calling loop.

    Args:
        query: The user's input query
        verbose: Whether to print intermediate steps

    Returns:
        Dictionary with input and output
    """
    ## TODO 4: Initialize the messages list with system prompt and user query
    ## Hint: Create a list with two dictionaries:
    ##   - First dict: role="system", content=SYSTEM_PROMPT
    ##   - Second dict: role="user", content=query
    messages = []

    if verbose:
        print(f"\n{'='*60}")
        print(f"User Query: {query}")
        print('='*60)

    # Agent loop - keep going until no more tool calls
    max_iterations = 10
    for i in range(max_iterations):
        # Get response from LLM
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        # Check if there are tool calls
        if not response.tool_calls:
            # No tool calls, we have the final answer
            if verbose:
                print(f"\nFinal Answer: {response.content}")
            return {"input": query, "output": response.content}

        # Process each tool call
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            if verbose:
                print(f"\n> Calling tool: {tool_name}")
                print(f"  Arguments: {tool_args}")

            ## TODO 3: Execute the tool and get the result
            ## Hint: Use tools_dict to get the tool by name, then call .invoke() with tool_args
            ## Don't forget to handle the case when tool_name is not in tools_dict
            if tool_name in tools_dict:
                tool_result = None  # Replace with the actual tool invocation
            else:
                tool_result = f"Error: Unknown tool {tool_name}"

            if verbose:
                print(f"  Result: {tool_result}")

            # Add tool result to messages
            messages.append(ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"]
            ))

    return {"input": query, "output": "Max iterations reached"}


if __name__ == "__main__":
    # Test your implementation with these queries:
    query = "I want to book a trip on 2023-12-25 to Paris, France. 2 people for 3 nights. Its a business trip"
    output = run_agent_with_query(query)

    query_2 = "What is the status of reservation 9c89a904?"
    output_2 = run_agent_with_query(query_2)
    print(output_2)
