import pandas as pd
import matplotlib.pyplot as plt


def safe_convert_list(row: str, data_type: str):
    """
    Safely convert a string representation of a list to an actual list,
    with type conversion tailored to specific data types including handling
    for 'states' which are extracted and stripped of surrounding quotes.
    """
    try:
        if data_type == 'jammer_position':
            result = row.strip('[').strip(']').split(', ')
            return [float(pos) for pos in result]
        elif data_type == 'node_positions':
            result = row.strip('[').strip(']').split('], [')
            return [[float(num) for num in elem.split(', ')] for elem in result]
        elif data_type == 'node_states':
            result = row.strip('[').strip(']').split(', ')
            return [int(state) for state in result]
        else:
            raise ValueError("Unknown data type")
    except (ValueError, SyntaxError, TypeError):
        return []


# Load the CSV
df = pd.read_csv('fspl/circle.csv')
# df = pd.read_csv('fspl/random_jammer_outside_region.csv')
# df = pd.read_csv('fspl/all_jammed.csv')

# Convert the columns
df['node_positions'] = df['node_positions'].apply(lambda x: safe_convert_list(x, 'node_positions'))
df['jammer_position'] = df['jammer_position'].apply(lambda x: safe_convert_list(x, 'jammer_position'))
df['node_states'] = df['node_states'].apply(lambda x: safe_convert_list(x, 'node_states'))

# Plotting
for index, row in df.iterrows():
    node_positions = row['node_positions']
    jammer_position = row['jammer_position']
    node_states = row['node_states']

    # Extract x and y coordinates for nodes
    node_x, node_y = zip(*node_positions)

    # Determine colors based on states (0 for unjammed, 1 for jammed)
    colors = ['red' if state == 1 else 'blue' for state in node_states]

    # Plot nodes with appropriate colors
    plt.scatter(node_x, node_y, c=colors, label='Nodes')

    # Plot jammer in black
    plt.scatter(jammer_position[0], jammer_position[1], color='black', label='Jammer', marker='x')

    plt.title(f'Instance {index + 1}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.grid(True)
    plt.show()
