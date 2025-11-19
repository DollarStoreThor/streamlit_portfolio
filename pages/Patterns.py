import streamlit as st

st.title("Quilted Pattern Evolution Game")
st.set_page_config(page_title="Andrew - Quilted Patterns",
                   layout="centered",
                   page_icon="🧵",
                   initial_sidebar_state="expanded")


st.subheader("Explore Quilted Patterns! 🧵", divider=True)


import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim


class quilted_pattern():
    def __init__(self, genome_length=100):
        self.GENOME = np.random.randint(2, size=genome_length)

        self.SPACING = 6
        self.SCALE = 2

        self.PATTERN_SIZE_X = int(4 * self.SCALE * self.SPACING)
        self.PATTERN_SIZE_Y = int(3 * self.SCALE * self.SPACING)

        self.FINAL_PATTERN = np.zeros((self.PATTERN_SIZE_X, self.PATTERN_SIZE_Y))
        self.MAJOR_PATTERN = np.zeros((self.SPACING*4, self.SPACING*3))
        self.COLOR_PATTERN = np.zeros((self.PATTERN_SIZE_X, self.PATTERN_SIZE_Y))

        self.MAJOR_PATTERN = self.create_major_pattern()
        self.MINOR_PATTERN = self.create_minor_pattern()

        self.DONE = self.calculate_final_pattern()

    def create_major_pattern(self):
        for coords, y in np.ndenumerate(self.MAJOR_PATTERN):
            if self.GENOME[99]:
                if (coords[0] + coords[1]) % self.SPACING == 0:
                    self.MAJOR_PATTERN[coords] = self.MAJOR_PATTERN[coords] + 3
                else:
                    self.MAJOR_PATTERN[coords] = self.MAJOR_PATTERN[coords] + 1

            if self.GENOME[98]:
                if (coords[0] - coords[1]) % self.SPACING == 0:
                    self.MAJOR_PATTERN[coords] = self.MAJOR_PATTERN[coords] +  3
                else:
                    self.MAJOR_PATTERN[coords] = self.MAJOR_PATTERN[coords] + 1

            if self.GENOME[97]:
                if (coords[0] * coords[1]) % self.SPACING == 0:
                    self.MAJOR_PATTERN[coords] = self.MAJOR_PATTERN[coords] +  3 
                else:
                    self.MAJOR_PATTERN[coords] = self.MAJOR_PATTERN[coords] + 1
            if self.GENOME[96]:
                if (coords[0] + 2 * coords[1]) % self.SPACING == 0:
                    self.MAJOR_PATTERN[coords] = self.MAJOR_PATTERN[coords] +  3
                else:
                    self.MAJOR_PATTERN[coords] = self.MAJOR_PATTERN[coords] + 1
            if self.GENOME[95]:
                if (self.SPACING * coords[0] - coords[1]*self.GENOME[10]) % self.SPACING == 0:
                    self.MAJOR_PATTERN[coords] = self.MAJOR_PATTERN[coords] +  3
                else:
                    self.MAJOR_PATTERN[coords] = self.MAJOR_PATTERN[coords] + 1
            if self.GENOME[94]:
                if (coords[0] ** self.SPACING * coords[1]) % self.SPACING == 0:
                    self.MAJOR_PATTERN[coords] = self.MAJOR_PATTERN[coords] +  3
                else:
                    self.MAJOR_PATTERN[coords] = self.MAJOR_PATTERN[coords] + 1
            if self.GENOME[93]:
                if (coords[0] + coords[1] ** self.SPACING) % self.SPACING == 0:
                    self.MAJOR_PATTERN[coords] = self.MAJOR_PATTERN[coords] +  3
                else:
                    self.MAJOR_PATTERN[coords] = self.MAJOR_PATTERN[coords] + 1
            if self.GENOME[92]:
                if (coords[0] - coords[1] ** self.SPACING) % self.SPACING == 0:
                    self.MAJOR_PATTERN[coords] = self.MAJOR_PATTERN[coords] +  3
                else:
                    self.MAJOR_PATTERN[coords] = self.MAJOR_PATTERN[coords] + 1
            if self.GENOME[91]:
                if (coords[0] * 3 + coords[1] * 2) % self.SPACING == 0:
                    self.MAJOR_PATTERN[coords] = self.MAJOR_PATTERN[coords] +  3
                else:
                    self.MAJOR_PATTERN[coords] = self.MAJOR_PATTERN[coords] + 1
            if self.GENOME[90]:
                if (coords[0] + coords[1] ** (2 + self.GENOME[80] + self.GENOME[52])) % self.SPACING % 2 == 0:
                    self.MAJOR_PATTERN[coords] = self.MAJOR_PATTERN[coords] +  3
                else:
                    self.MAJOR_PATTERN[coords] = self.MAJOR_PATTERN[coords] + 1
                    
        return self.MAJOR_PATTERN

    def create_minor_pattern(self):
        for coords, _ in np.ndenumerate(self.FINAL_PATTERN):
            if self.GENOME[0]:
                if (coords[0] + coords[1]) % self.SCALE * self.SPACING == 0:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] + 6 * self.GENOME[16]
                else:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] + 1

            if self.GENOME[1]:
                if (coords[0] - coords[1]) % self.SPACING == 0:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] + 3
                else:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] + 1

            if self.GENOME[2]:
                if (coords[0] * coords[1]) % self.SPACING == 0:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] + 3
                else:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] + 1

            if self.GENOME[3]:
                if (coords[1] + coords[0] * self.SPACING) % self.SPACING == 0:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] + 3
                else:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] + 1

            if self.GENOME[4]:
                if (coords[0] + coords[1]*self.GENOME[3]) % self.SPACING == 0:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] +  3
                else:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] + 1

            if self.GENOME[5]:
                if (coords[1] == coords[0] * (coords[1] + 1)) % self.SPACING == 0:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] +  3
                else:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] + 1

            if self.GENOME[6]:
                if (coords[0] + coords[1] ** 2) % self.SPACING == 0:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] +  3
                else:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] + 1
            
            if self.GENOME[7]:
                if (coords[0] ** 2 - coords[1]) % self.SPACING == 0:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] +  3
                else:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] + 1

            if self.GENOME[8]:
                if (coords[0] * 2 + coords[1] * 3) % self.SPACING == 0:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] +  6 * self.GENOME[15]
                else:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] + 1

            if self.GENOME[9]:
                if (coords[0] ** 2 + coords[1] ** 2) % self.SPACING == 0:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] +  3
                else:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] + 1

            if self.GENOME[10]:
                if (coords[0] * coords[1] + coords[0] - coords[1] ** self.GENOME[5]) % self.SPACING == 0:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] +  3
                else:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] + 1

            if self.GENOME[11]:
                if (coords[0] - coords[1] ** (2*self.GENOME[14])) % self.SPACING == 0:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] +  3
                else:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] + 1

            if self.GENOME[12]:
                if (coords[0] + coords[1] * self.GENOME[8] * self.GENOME[9]) % self.SPACING % 2 == 0:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] +  3
                else:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] + 1

            if self.GENOME[13]:
                if (coords[0] ** 3 + coords[1]) % self.SPACING == 0:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] +  3
                else:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] + 1
                    
            if self.GENOME[14]:
                if ( coords[1]**3) % self.SPACING != 0:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] +  3
                else:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] + 1

            if self.GENOME[15]:
                if (coords[0] - coords[1] * self.GENOME[7]) % self.SPACING == 0:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] +  3
                else:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] + 1

            if self.GENOME[16]:
                if (coords[0] * self.GENOME[51] + coords[1] * self.GENOME[31]) % self.SPACING == 0:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] +  3
                else:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] + 1

            if self.GENOME[17]:
                if (coords[0]**2 + coords[1]**3) % self.SPACING != 0:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] +  3
                else:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] + 1 

            if self.GENOME[18]:
                if self.GENOME[20] == 1:
                    # invert pattern
                    if self.FINAL_PATTERN[coords] == 0:
                        self.FINAL_PATTERN[coords] = self.FINAL_PATTERN.max() * 6

            if self.GENOME[19]:
                if (coords[0] + coords[1] + self.GENOME[22]) % self.SPACING == 0:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] +  3
                else:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] + 1

            if self.GENOME[20]:
                if (coords[0] * self.GENOME[33] - coords[1] * self.GENOME[44]) % self.SPACING == self.GENOME[92]:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] +  6 * self.GENOME[25]
                else:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] + 1

            if self.GENOME[21]:
                if (coords[0]**3 - coords[1]**2) % self.SPACING == self.GENOME[31]:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] +  3
                else:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] + 1

            if self.GENOME[22] and self.GENOME[23]:
                if (coords[0] + coords[1] + self.GENOME[10]) % self.SPACING == 0:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] +  10 * self.GENOME[30]

            if self.GENOME[24]:
                if (coords[0]**4 + coords[1]**4) % (self.SPACING**2) == 8:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] + self.SPACING*2

            if self.GENOME[25]:
                if(coords[0]**3 + coords[1]**4) % self.SPACING**(1+self.GENOME[9]) == 2:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] + (np.sum(self.GENOME) // np.max(self.FINAL_PATTERN))
                    
            if self.GENOME[26]:
                if(coords[0]**4 - coords[1]**4) % self.SPACING == 0:
                    self.FINAL_PATTERN[coords] = self.FINAL_PATTERN[coords] + (np.sum(self.GENOME) // np.max(self.FINAL_PATTERN))

        return self.FINAL_PATTERN

    def calculate_final_pattern(self):
        if not np.array_equal(self.GENOME[90:], np.zeros(10)):
            DONE = self.FINAL_PATTERN * np.repeat(np.repeat(self.MAJOR_PATTERN, self.SCALE, axis=0), self.SCALE, axis=1)
        else:
            DONE = self.FINAL_PATTERN
        return DONE

def breed(quilted_pattern_1, quilted_pattern_2, num_children=2, num_chromosomes=1, verbose=False):

    quilted_pattern_1_genome = quilted_pattern_1.GENOME
    quilted_pattern_2_genome = quilted_pattern_2.GENOME

    GENOME_LENGTH = len(quilted_pattern_1_genome)
    
    children = []
    for child in range(num_children):

        # Calculate chromosome length
        chromosome_length = GENOME_LENGTH // num_chromosomes
        
        # Initialize child genomes
        child_1_genome = quilted_pattern_1_genome.copy()
        child_2_genome = quilted_pattern_2_genome.copy()
        
        # Perform crossover for each chromosome
        for i in range(num_chromosomes):
            # Determine chromosome boundaries
            chromosome_start = i * chromosome_length
            chromosome_end = (i + 1) * chromosome_length if i < num_chromosomes - 1 else GENOME_LENGTH
            
            # Random crossover point within this chromosome
            crossover_point = np.random.randint(chromosome_start + 1, chromosome_end)
            if verbose:
                print(f"{child+1}/{num_children} - ", end="")
                print(f"Crossover at chromosome {i}, point {crossover_point}")

            # Swap segments within this chromosome
            child_1_genome[crossover_point:chromosome_end] = quilted_pattern_2_genome[crossover_point:chromosome_end]
            child_2_genome[crossover_point:chromosome_end] = quilted_pattern_1_genome[crossover_point:chromosome_end]
        
        child_1 = quilted_pattern()
        child_2 = quilted_pattern()

        child_1.GENOME = child_1_genome
        child_2.GENOME = child_2_genome

        child_1.DONE = child_1.calculate_final_pattern()
        child_2.DONE = child_2.calculate_final_pattern()

        if np.random.rand() < 0.5:
            children.append(child_1)
        else:
            children.append(child_2)

    return children

def mutate(quilted_pattern, mutation_rate=0.01, verbose=False):
    for i in range(len(quilted_pattern.GENOME)):
        if np.random.rand() < mutation_rate:
            if verbose:
                print(f"Mutating gene index {i} from {quilted_pattern.GENOME[i]} to {abs(1 - quilted_pattern.GENOME[i])}")
            quilted_pattern.GENOME[i] = abs(1 - quilted_pattern.GENOME[i])  # Flip bit
    quilted_pattern.DONE = quilted_pattern.calculate_final_pattern()
    return quilted_pattern

# Compute the similarity between two patterns
def pattern_similarity(pattern1, pattern2):
    # Determine color maps based on genome segments
    color_map_index_1 = pattern1.GENOME[40:60].sum() % len(color_map)
    color_map_index_2 = pattern2.GENOME[40:60].sum() % len(color_map)

    # Create figure for pattern1
    fig1, ax1 = plt.subplots(figsize=(10, 10), dpi=100)
    ax1.imshow(pattern1.DONE, cmap=color_map[color_map_index_1])
    ax1.axis('off')
    
    # Render to array
    fig1.canvas.draw()
    pattern1_image = np.frombuffer(fig1.canvas.buffer_rgba(), dtype=np.uint8)
    pattern1_image = pattern1_image.reshape(fig1.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig1)

    # Create figure for pattern2
    fig2, ax2 = plt.subplots(figsize=(10, 10), dpi=100)
    ax2.imshow(pattern2.DONE, cmap=color_map[color_map_index_2])
    ax2.axis('off')

    # Render to array
    fig2.canvas.draw()
    pattern2_image = np.frombuffer(fig2.canvas.buffer_rgba(), dtype=np.uint8)
    pattern2_image = pattern2_image.reshape(fig2.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig2)

    # Normalize patterns to ensure they're in a valid range
    min_val = min(pattern1_image.min(), pattern2_image.min())
    max_val = max(pattern1_image.max(), pattern2_image.max())
    
    # Handle edge case where patterns are constant
    if max_val == min_val:
        data_range = 1.0
    else:
        data_range = max_val - min_val
    
    # Determine appropriate win_size based on image dimensions
    min_dim = min(pattern1_image.shape[0], pattern1_image.shape[1])
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    
    # Compute structural similarity
    similarity_score, _ = ssim(pattern1_image, pattern2_image, full=True, data_range=data_range, win_size=win_size, channel_axis=2)

    return similarity_score

def founder_population(population_size=8):
    population = []
    for _ in range(population_size):
        population.append(quilted_pattern())
    return population

def selection(population, verbose=False):
    print("Select two parents from the following population:")
    Parent_1_choice = int(input("Select Parent 1: (ENTER THE INDEX)"))
    Parent_2_choice = int(input("Select Parent 2: (ENTER THE INDEX)"))

    #ensure user selects valid parents
    if Parent_1_choice < 1 or Parent_1_choice > len(population) or Parent_2_choice < 1 or Parent_2_choice > len(population):
        print("Invalid selection. Please select valid parent indices.")
        return selection(population)
    
    if verbose:
        print("Pattern Similarity Between Bred Patterns:")
        print(pattern_similarity(population[Parent_1_choice - 1], population[Parent_2_choice - 1]))

    return population[Parent_1_choice-1], population[Parent_2_choice-1]

color_map = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
                      'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
                      'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']

def display_population(population, item_to_match):
    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    axes = axes.flatten()
    for i in range(len(population)):
        axes[i].set_title(f'Quilt {i+1}')
        color_map_index = population[i].GENOME[40:60].sum() % len(color_map)
        axes[i].imshow(population[i].DONE, cmap=color_map[color_map_index])
        axes[i].axis('off')
    plt.tight_layout()
    st.pyplot(fig)

    # visulaize the item_to_match
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_title('Item to Match')
    color_map_index = item_to_match.GENOME[40:60].sum() % len(color_map)
    ax.imshow(item_to_match.DONE, cmap=color_map[color_map_index])
    ax.axis('off')
    st.pyplot(fig)

def game_loop(verbose=False):

    # Founder Population
    population = founder_population()
    item_to_match = quilted_pattern()

    # Display Founder Population
    display_population(population, item_to_match)

    st.write("Evolve your patterns to match the target pattern!")
    generations = 1

    # Human Selection and Breeding 
    continue_evolving = True
    while continue_evolving:
        st.write(f"Generation: {generations}")
        selected = selection(population=population)
        new_population = breed(selected[0], selected[1], num_children=len(population), num_chromosomes=5)

        # Force Mutations on Genration 1
        if generations == 1:
            introduce_mutation = 'y'
        # Ask User if they want to introduce mutations on subsequent generations
        else:
            introduce_mutation = st.chat_input("Introduce mutation? (y/n)")

        # Apply mutations if user chooses to
        if introduce_mutation.lower() == 'y':
            for i in range(len(new_population)):
                new_population[i] = mutate(new_population[i], mutation_rate=0.005)

        # Update population and generation count
        population = new_population
        generations += 1

        # Display current population's similarity scores (if verbose == True)
        if verbose:
            for item in population:
                st.write("Pattern Similarity to Item to Match:", pattern_similarity(item, item_to_match))
        
        # Draw current population
        display_population(population, item_to_match)

        # Ask user if they want to continue evolving
        continue_evolving = st.chat_input("Continue evolution? (y/n)") == 'y'

    # Final Selection for the User to choose their best pattern
    user_choice_index = -1
    while user_choice_index < 1 or user_choice_index > len(population):
            user_choice_index = int(st.chat_input(f'Select your final pattern choice (1-{len(population)}): '))
            if 1 <= user_choice_index <= len(population):
                break
            else:
                st.write(f"Please enter a number between 1 and {len(population)}.")

    final_user_choice = population[user_choice_index - 1]
    st.write(pattern_similarity(final_user_choice, item_to_match))
    st.write("Final score: ", (pattern_similarity(final_user_choice, item_to_match) // ((generations + 99) / 100000)))

with st.container():
    game_loop()