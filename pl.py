import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, clear_output
import ipywidgets as widgets
from matplotlib.colors import LinearSegmentedColormap
import random
from tqdm.notebook import tqdm
import matplotlib.ticker as ticker

class PremierLeagueSimulation:
    def __init__(self):
            # Team names in the same order as the provided data
            self.team_names = [
    'Liverpool', 'Man_City', 'Arsenal', 'Man_Utd', 'Chelsea', 
    'Tottenham', 'NewCastle', 'Aston_Villa', 'West_Ham', 'Brighton',
    'Nottingham_Forest', 'Fulham', 'AFC_Bournemouth', 'Brentford', 
    'Crystal_Palace', 'Everton', 'Wolves', 'Ipswich_Town', 
    'Leicester_City', 'Southampton'
]

# Last three years of xGF data (most recent first)
xgf_data = [
    [2.08, 2.03, 1.83, 1.51, 1.6, 1.73, 1.57, 1.46, 1.3, 1.59, 
     1.28, 1.44, 1.51, 1.39, 1.32, 1.43, 1.27, 1.59, 1.6, 1.65],
    [1.89, 1.96, 1.94, 1.77, 1.61, 1.62, 1.77, 1.38, 1.51, 1.94, 
     1.18, 1.42, 1.18, 1.42, 1.34, 1.39, 1.32, 1.96, 1.34, 1.32],
    [2.29, 2.36, 1.86, 1.69, 1.95, 1.65, 1.43, 1.50, 1.56, 1.58, 
     1.5, 1.92, 1.58, 1.47, 1.41, 1.44, 1.37, 1.46, 1.46, 1.57]
]

# Last three years of xGA data (most recent first)
xga_data = [
    [1.23, 0.93, 0.92, 1.73, 1.51, 1.38, 1.51, 1.38, 1.85, 1.33, 
     1.52, 1.56, 1.54, 1.62, 1.38, 1.47, 1.65, 1.18, 1.12, 1.15],
    [1.29, 0.99, 1.21, 1.49, 1.48, 1.64, 1.32, 1.47, 1.60, 1.27, 
     1.76, 1.65, 1.96, 1.77, 1.55, 1.75, 1.70, 0.94, 1.65, 1.60],
    [1.08, 0.89, 1.41, 1.65, 1.20, 1.57, 1.72, 1.54, 1.71, 1.54, 
     1.38, 1.15, 1.27, 1.71, 1.51, 1.77, 1.70, 1.14, 1.81, 1.67]
]

            # Calculate weighted averages (more weight to recent seasons)
            weights = [0.5, 0.3, 0.2]  # 50% last year, 30% two years ago, 20% three years ago
            
            # Calculate weighted xGF and xGA for each team
            weighted_xgf = {}
            weighted_xga = {}
            
            for idx, team in enumerate(self.team_names):
                team_xgf = sum(xgf_data[year_idx][idx] * weights[year_idx] for year_idx in range(3))
                team_xga = sum(xga_data[year_idx][idx] * weights[year_idx] for year_idx in range(3))
                
                weighted_xgf[team] = team_xgf
                weighted_xga[team] = team_xga
            
            # Calculate average xGF and xGA across all teams
            avg_xgf = sum(weighted_xgf.values()) / len(self.team_names)
            avg_xga = sum(weighted_xga.values()) / len(self.team_names)
            
            # Calculate attack and defense ratings
            # Higher attack = better than average xGF
            # Higher defense = better than average xGA (lower xGA)
            self.teams = {}
            
            for team in self.team_names:
                attack_rating = (weighted_xgf[team] / avg_xgf) * 80  # Scale to roughly 0-100
                defense_rating = (avg_xga / weighted_xga[team]) * 80  # Inverse ratio for defense
                
                # Team colors (keeping the original colors)
                team_colors = {
                    'Liverpool': '#C8102E',
                    'Man_City': '#6CABDD',
                    'Arsenal': '#EF0107',
                    'Man_Utd': '#DA291C',
                    'Chelsea': '#034694',
                    'Tottenham': '#132257',
                    'NewCastle': '#241F20',
                    'Aston_Villa': '#95BFE5',
                    'West_Ham': '#7A263A',
                    'Brighton': '#0057B8'
                }
                
                # Estimate home advantage based on historical data
                # Using a fixed value for all teams, could be refined with more data
                home_advantage = 12
                
                self.teams[team] = {
                    'attack': attack_rating,
                    'defense': defense_rating,
                    'home_advantage': home_advantage,
                    'color': team_colors.get(team, '#000000'),
                    'xgf': weighted_xgf[team],  # Store the weighted xGF for reference
                    'xga': weighted_xga[team]   # Store the weighted xGA for reference
                }
            self.num_teams = len(self.team_names)
            self.standings = None
            self.matches = None
            self.all_simulations = []
            self.simulations_count = 0
            self.match_results = {}
    
    def generate_fixtures(self):
        """Generate a double round-robin schedule (home and away matches)"""
        fixtures = []
        
        # First half of the season (each team plays every other team once)
        for home_idx, home_team in enumerate(self.team_names):
            for away_idx, away_team in enumerate(self.team_names):
                if home_idx != away_idx:  # Teams don't play against themselves
                    fixtures.append({
                        'home_team': home_team,
                        'away_team': away_team,
                        'played': False,
                        'home_score': None,
                        'away_score': None,
                        'winner': None
                    })
        
        # Randomize the order of fixtures to simulate a realistic schedule
        random.shuffle(fixtures)
        
        # Initialize the standings DataFrame
        standings = pd.DataFrame({
            'Team': self.team_names,
            'Played': 0,
            'Won': 0,
            'Drawn': 0,
            'Lost': 0,
            'GF': 0,  # Goals For
            'GA': 0,  # Goals Against
            'GD': 0,  # Goal Difference
            'Points': 0
        })
        
        self.fixtures = fixtures
        self.standings = standings
        return fixtures

    def simulate_match(self, home_team, away_team):
        """Simulate a single match between two teams based on xG data"""
        # Get team ratings
        home_attack = self.teams[home_team]['attack']
        home_defense = self.teams[home_team]['defense']
        home_advantage = self.teams[home_team]['home_advantage']
        
        away_attack = self.teams[away_team]['attack']
        away_defense = self.teams[away_team]['defense']

        avg_defense = sum(team_data['defense'] for team_data in self.teams.values()) / self.num_teams

        # Calculate expected goals based on actual xG statistics
        # Home team gets a boost from home advantage
        lambda_home = self.teams[home_team]['xgf'] * (1 + home_advantage/100)
        lambda_away = self.teams[away_team]['xgf'] * 0.85  # Away teams tend to score less
        
        # Adjust based on the opponent's defensive strength
        lambda_home = lambda_home * (avg_defense / self.teams[away_team]['defense'])
        lambda_away = lambda_away * (avg_defense / self.teams[home_team]['defense'])
        
        # Add more unpredictability
        lambda_home = np.random.normal(lambda_home, 0.3)
        lambda_away = np.random.normal(lambda_away, 0.3)
        
        # Ensure non-negative values
        lambda_home = max(0.1, lambda_home)
        lambda_away = max(0.1, lambda_away)
        
        # Generate goals using Poisson distribution
        home_goals = np.random.poisson(lambda_home)
        away_goals = np.random.poisson(lambda_away)
        
        # Determine winner
        if home_goals > away_goals:
            winner = home_team
        elif away_goals > home_goals:
            winner = away_team
        else:
            winner = 'Draw'
        
        return home_goals, away_goals, winner
    
    # Calculate average defense for use in match simulation
  #  @property
   # def avg_defense(self):
    #    return sum(team_data['defense'] for team_data in self.teams.values()) / self.num_teams

    
    def simulate_season(self):
        """Simulate a complete season"""
        # Reset standings and generate new fixtures
        self.generate_fixtures()
        
        # Simulate each match
        for match in self.fixtures:
            home_team = match['home_team']
            away_team = match['away_team']
            
            home_goals, away_goals, winner = self.simulate_match(home_team, away_team)
            
            # Update match details
            match['played'] = True
            match['home_score'] = home_goals
            match['away_score'] = away_goals
            match['winner'] = winner
            
            # Update standings
            home_idx = self.standings.index[self.standings['Team'] == home_team].tolist()[0]
            away_idx = self.standings.index[self.standings['Team'] == away_team].tolist()[0]
            
            # Update matches played
            self.standings.at[home_idx, 'Played'] += 1
            self.standings.at[away_idx, 'Played'] += 1
            
            # Update goals
            self.standings.at[home_idx, 'GF'] += home_goals
            self.standings.at[home_idx, 'GA'] += away_goals
            self.standings.at[away_idx, 'GF'] += away_goals
            self.standings.at[away_idx, 'GA'] += home_goals
            
            # Update goal difference
            self.standings.at[home_idx, 'GD'] = self.standings.at[home_idx, 'GF'] - self.standings.at[home_idx, 'GA']
            self.standings.at[away_idx, 'GD'] = self.standings.at[away_idx, 'GF'] - self.standings.at[away_idx, 'GA']
            
            # Update W/D/L
            if winner == home_team:
                self.standings.at[home_idx, 'Won'] += 1
                self.standings.at[away_idx, 'Lost'] += 1
                self.standings.at[home_idx, 'Points'] += 3
            elif winner == away_team:
                self.standings.at[away_idx, 'Won'] += 1
                self.standings.at[home_idx, 'Lost'] += 1
                self.standings.at[away_idx, 'Points'] += 3
            else:  # Draw
                self.standings.at[home_idx, 'Drawn'] += 1
                self.standings.at[away_idx, 'Drawn'] += 1
                self.standings.at[home_idx, 'Points'] += 1
                self.standings.at[away_idx, 'Points'] += 1
                
            # Store the match result for later analysis
            if (home_team, away_team) not in self.match_results:
                self.match_results[(home_team, away_team)] = []
            
            self.match_results[(home_team, away_team)].append((home_goals, away_goals))
        
        # Sort the standings by points, then goal difference, then goals scored
        self.standings = self.standings.sort_values(by=['Points', 'GD', 'GF'], ascending=False).reset_index(drop=True)
        
        # Add position column
        self.standings['Position'] = range(1, len(self.standings) + 1)
        
        # Store for later analysis
        simulation_result = self.standings.copy()
        return simulation_result
    
    def run_monte_carlo(self, num_simulations=1000):
        """Run multiple simulations and store the results"""
        print(f"Running {num_simulations} Monte Carlo simulations...")
        self.all_simulations = []
        self.probability_matrix = np.zeros((self.num_teams, self.num_teams))
        self.champion_count = {team: 0 for team in self.team_names}
        self.top_four_count = {team: 0 for team in self.team_names}
        self.relegation_count = {team: 0 for team in self.team_names}  # Bottom 3 teams
        
        # Reset match results
        self.match_results = {}
        
        for _ in tqdm(range(num_simulations)):
            # Simulate a season
            season_result = self.simulate_season()
            self.all_simulations.append(season_result)
            
            # Count champion, top 4, relegation outcomes
            champion = season_result.iloc[0]['Team']
            self.champion_count[champion] += 1
            
            for i in range(4):
                if i < len(season_result):
                    top_four_team = season_result.iloc[i]['Team']
                    self.top_four_count[top_four_team] += 1
            
            for i in range(self.num_teams - 3, self.num_teams):
                if i < len(season_result):
                    relegated_team = season_result.iloc[i]['Team']
                    self.relegation_count[relegated_team] += 1
            
            # Fill the probability matrix
            for i, team in enumerate(self.team_names):
                position = season_result[season_result['Team'] == team]['Position'].values[0] - 1
                self.probability_matrix[i, position] += 1
        
        # Normalize probability matrix
        self.probability_matrix = self.probability_matrix / num_simulations
        
        # Calculate average position for each team
        self.avg_positions = {}
        for i, team in enumerate(self.team_names):
            positions = [s[s['Team'] == team]['Position'].values[0] for s in self.all_simulations]
            self.avg_positions[team] = sum(positions) / len(positions)
        
        # Calculate average expected points
        self.avg_points = {}
        for team in self.team_names:
            points = [s[s['Team'] == team]['Points'].values[0] for s in self.all_simulations]
            self.avg_points[team] = sum(points) / len(points)
        
        self.simulations_count = num_simulations
        print(f"Completed {num_simulations} simulations.")
    
    def show_average_table(self):
        """Show average standings from all simulations"""
        if not self.all_simulations:
            print("No simulations have been run yet.")
            return
        
        # Calculate average stats for each team
        avg_stats = {team: {
            'Played': 0,
            'Won': 0,
            'Drawn': 0,
            'Lost': 0,
            'GF': 0,
            'GA': 0,
            'GD': 0,
            'Points': 0
        } for team in self.team_names}
        
        for sim in self.all_simulations:
            for _, row in sim.iterrows():
                team = row['Team']
                for stat in ['Played', 'Won', 'Drawn', 'Lost', 'GF', 'GA', 'GD', 'Points']:
                    avg_stats[team][stat] += row[stat]
        
        # Divide by number of simulations
        for team in self.team_names:
            for stat in avg_stats[team]:
                avg_stats[team][stat] /= len(self.all_simulations)
        
        # Create a DataFrame from average stats
        avg_table = pd.DataFrame({
            'Team': list(avg_stats.keys()),
            'Avg Position': [self.avg_positions[team] for team in avg_stats],
            'Avg Points': [avg_stats[team]['Points'] for team in avg_stats],
            'Avg Won': [avg_stats[team]['Won'] for team in avg_stats],
            'Avg Drawn': [avg_stats[team]['Drawn'] for team in avg_stats],
            'Avg Lost': [avg_stats[team]['Lost'] for team in avg_stats],
            'Avg GF': [avg_stats[team]['GF'] for team in avg_stats],
            'Avg GA': [avg_stats[team]['GA'] for team in avg_stats],
            'Avg GD': [avg_stats[team]['GD'] for team in avg_stats],
            'Champion %': [self.champion_count[team]/self.simulations_count*100 for team in avg_stats],
            'Top 4 %': [self.top_four_count[team]/self.simulations_count*100 for team in avg_stats],
            'Relegation %': [self.relegation_count[team]/self.simulations_count*100 for team in avg_stats]
        })
        
        # Sort by average position
        avg_table = avg_table.sort_values('Avg Position').reset_index(drop=True)
        
        # Format
        for col in avg_table.columns:
            if 'Avg' in col and col != 'Avg Position':
                avg_table[col] = avg_table[col].round(1)
            elif '%' in col:
                avg_table[col] = avg_table[col].round(1)
        
        avg_table['Avg Position'] = avg_table['Avg Position'].round(2)
        
        return avg_table
    
    def plot_position_probabilities(self, figsize=(12, 8)):
        """Plot heatmap of team position probabilities"""
        if not hasattr(self, 'probability_matrix'):
            print("No simulations have been run yet.")
            return
        
        # Sort teams by average position
        sorted_teams = [team for team in sorted(self.avg_positions.items(), key=lambda x: x[1])]
        sorted_team_names = [team[0] for team in sorted_teams]
        
        plt.figure(figsize=figsize)
        
        # Reorder probability matrix according to average position
        ordered_matrix = np.zeros_like(self.probability_matrix)
        for i, team in enumerate(sorted_team_names):
            original_idx = self.team_names.index(team)
            ordered_matrix[i] = self.probability_matrix[original_idx]
        
        # Custom colormap: From white to blue
        custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#FFFFFF', '#0057B8'], N=100)
        
        # Plot the heatmap
        ax = plt.gca()
        im = ax.imshow(ordered_matrix * 100, cmap=custom_cmap, aspect='auto', vmin=0, vmax=60)
        
        # Set labels
        positions = [f"{i+1}" for i in range(self.num_teams)]
        plt.xticks(np.arange(len(positions)), positions)
        plt.yticks(np.arange(len(sorted_team_names)), sorted_team_names)
        
        # Add colorbar
        cbar = plt.colorbar(im)
        cbar.set_label('Probability (%)')
        
        # Add text annotations
        for i in range(len(sorted_team_names)):
            for j in range(len(positions)):
                prob = ordered_matrix[i, j] * 100
                if prob >= 5:  # Only show significant probabilities to avoid clutter
                    text_color = 'black' if prob < 40 else 'white'
                    ax.text(j, i, f"{prob:.1f}%", ha="center", va="center", color=text_color)
        
        plt.title('Probability (%) of Each Team Finishing in a Specific Position', fontsize=14)
        plt.xlabel('Position', fontsize=12)
        plt.tight_layout()
        
        return plt
    
    def plot_champion_probabilities(self, figsize=(10, 6)):
        """Plot champion probabilities"""
        if not self.all_simulations:
            print("No simulations have been run yet.")
            return
        
        # Sort teams by champion probability
        sorted_teams = sorted(self.champion_count.items(), key=lambda x: x[1], reverse=True)
        teams = [team[0] for team in sorted_teams]
        probs = [team[1]/self.simulations_count*100 for team in sorted_teams]
        
        # Get team colors
        colors = [self.teams[team]['color'] for team in teams]
        
        # Create the bar chart
        plt.figure(figsize=figsize)
        bars = plt.bar(teams, probs, color=colors)
        
        # Add percentage labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom')
        
        plt.title('Probability (%) of Winning the Premier League', fontsize=14)
        plt.ylabel('Probability (%)')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, max(probs) * 1.2)  # Set y limit with some headroom
        plt.tight_layout()
        
        return plt
    
    def plot_top_four_probabilities(self, figsize=(10, 6)):
        """Plot top four probabilities"""
        if not self.all_simulations:
            print("No simulations have been run yet.")
            return
        
        # Sort teams by top four probability
        sorted_teams = sorted(self.top_four_count.items(), key=lambda x: x[1], reverse=True)
        teams = [team[0] for team in sorted_teams]
        probs = [team[1]/self.simulations_count*100 for team in sorted_teams]
        
        # Get team colors
        colors = [self.teams[team]['color'] for team in teams]
        
        # Create the bar chart
        plt.figure(figsize=figsize)
        bars = plt.bar(teams, probs, color=colors)
        
        # Add percentage labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom')
        
        plt.title('Probability (%) of Finishing in the Top 4', fontsize=14)
        plt.ylabel('Probability (%)')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, max(probs) * 1.2)  # Set y limit with some headroom
        plt.tight_layout()
        
        return plt
    
    def plot_relegation_probabilities(self, figsize=(10, 6)):
        """Plot relegation probabilities (bottom 3)"""
        if not self.all_simulations:
            print("No simulations have been run yet.")
            return
        
        # Sort teams by relegation probability
        sorted_teams = sorted(self.relegation_count.items(), key=lambda x: x[1], reverse=True)
        teams = [team[0] for team in sorted_teams]
        probs = [team[1]/self.simulations_count*100 for team in sorted_teams]
        
        # Get team colors
        colors = [self.teams[team]['color'] for team in teams]
        
        # Create the bar chart
        plt.figure(figsize=figsize)
        bars = plt.bar(teams, probs, color=colors)
        
        # Add percentage labels
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom')
        
        plt.title('Probability (%) of Finishing in the Bottom 3', fontsize=14)
        plt.ylabel('Probability (%)')
        plt.xticks(rotation=45, ha='right')
        plt.ylim(0, max(probs) * 1.2)  # Set y limit with some headroom
        plt.tight_layout()
        
        return plt
    
    def plot_expected_points_distribution(self, figsize=(12, 8)):
        """Plot the distribution of expected points for each team"""
        if not self.all_simulations:
            print("No simulations have been run yet.")
            return
        
        # Collect points data for each team across all simulations
        points_data = {team: [] for team in self.team_names}
        for sim in self.all_simulations:
            for _, row in sim.iterrows():
                team = row['Team']
                points = row['Points']
                points_data[team].append(points)
        
        # Sort teams by average points
        teams_by_avg_points = sorted(self.avg_points.items(), key=lambda x: x[1], reverse=True)
        sorted_teams = [team[0] for team in teams_by_avg_points]
        
        # Determine plotting limits
        all_points = [p for team_points in points_data.values() for p in team_points]
        min_points = min(all_points)
        max_points = max(all_points)
        
        # Set up the plot
        plt.figure(figsize=figsize)
        
        # Plot violin plots for each team's points distribution
        violin_parts = plt.violinplot(
            [points_data[team] for team in sorted_teams],
            showmeans=True,
            showmedians=False,
            vert=False
        )
        
        # Color the violin plots
        for i, team in enumerate(sorted_teams):
            for pc in violin_parts['bodies']:
                pc.set_facecolor(self.teams[team]['color'])
                pc.set_alpha(0.7)
        
        # Add team names
        plt.yticks(np.arange(1, len(sorted_teams) + 1), sorted_teams)
        
        # Add grid lines
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        
        # Add axis labels and title
        plt.xlabel('Points', fontsize=12)
        plt.title('Expected Points Distribution by Team', fontsize=14)
        
        # Format x-axis
        plt.xlim(min_points - 2, max_points + 2)
        
        # Add mean points as text
        for i, team in enumerate(sorted_teams):
            avg_points = self.avg_points[team]
            plt.text(
                avg_points + 1, i + 1,
                f"{avg_points:.1f}",
                va='center'
            )
        
        plt.tight_layout()
        return plt
    
    def plot_head_to_head(self, team1, team2, figsize=(10, 6)):
        """Plot head-to-head results between two teams"""
        if not self.match_results:
            print("No simulations have been run yet.")
            return
        
        match_key1 = (team1, team2)
        match_key2 = (team2, team1)
        
        if match_key1 not in self.match_results and match_key2 not in self.match_results:
            print(f"No match data found for {team1} vs {team2}")
            return
        
        # Collect all results
        team1_home_results = self.match_results.get(match_key1, [])
        team2_home_results = self.match_results.get(match_key2, [])
        
        team1_wins = sum(1 for goals1, goals2 in team1_home_results if goals1 > goals2)
        team2_wins = sum(1 for goals1, goals2 in team2_home_results if goals2 > goals1)
        draws_team1_home = sum(1 for goals1, goals2 in team1_home_results if goals1 == goals2)
        draws_team2_home = sum(1 for goals1, goals2 in team2_home_results if goals1 == goals2)
        
        total_matches = len(team1_home_results) + len(team2_home_results)
        team1_total_wins = team1_wins + sum(1 for goals1, goals2 in team2_home_results if goals2 > goals1)
        team2_total_wins = team2_wins + sum(1 for goals1, goals2 in team1_home_results if goals1 > goals2)
        total_draws = draws_team1_home + draws_team2_home
        
        # Create the plot
        plt.figure(figsize=figsize)
        
        # Create pie chart
        labels = [f"{team1} Wins", f"{team2} Wins", "Draws"]
        sizes = [team1_total_wins, team2_total_wins, total_draws]
        colors = [self.teams[team1]['color'], self.teams[team2]['color'], 'gray']
        
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, wedgeprops={'alpha': 0.8})
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
        
        # Add additional text information
        plt.title(f"Head-to-Head: {team1} vs {team2} ({total_matches} matches)", fontsize=14)
        
        # Add text for home/away breakdown
        text_info = [
            f"{team1} Home: {team1_wins}W {draws_team1_home}D {len(team1_home_results) - team1_wins - draws_team1_home}L",
            f"{team2} Home: {team2_wins}W {draws_team2_home}D {len(team2_home_results) - team2_wins - draws_team2_home}L"
        ]
        
        plt.figtext(0.5, 0.01, "\n".join(text_info), ha="center", fontsize=10, 
                    bbox={"facecolor":"white", "alpha":0.5, "pad":5})
        
        plt.tight_layout()
        return plt
    
    def simulate_single_match_interactive(self, home_team, away_team, num_simulations=1000):
        """Simulate a single match multiple times and show the distribution of results"""
        if home_team == away_team:
            print("Teams must be different.")
            return
        
        home_wins = 0
        away_wins = 0
        draws = 0
        home_goals_total = 0
        away_goals_total = 0
        
        home_score_dist = {}
        away_score_dist = {}
        results = []
        
        for _ in range(num_simulations):
            home_goals, away_goals, winner = self.simulate_match(home_team, away_team)
            
            results.append((home_goals, away_goals))
            
            if winner == home_team:
                home_wins += 1
            elif winner == away_team:
                away_wins += 1
            else:
                draws += 1
                
            home_goals_total += home_goals
            away_goals_total += away_goals
            
            # Update score distributions
            score_key = f"{home_goals}-{away_goals}"
            if score_key not in home_score_dist:
                home_score_dist[score_key] = 0
            home_score_dist[score_key] += 1
        
        # Convert counts to percentages
        for key in home_score_dist:
            home_score_dist[key] = (home_score_dist[key] / num_simulations) * 100
            
        # Sort by frequency
        sorted_scores = sorted(home_score_dist.items(), key=lambda x: x[1], reverse=True)
        
        # Prepare plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Pie chart for match outcomes
        labels = [f"{home_team} Win", f"{away_team} Win", "Draw"]
        sizes = [home_wins/num_simulations*100, away_wins/num_simulations*100, draws/num_simulations*100]
        colors = [self.teams[home_team]['color'], self.teams[away_team]['color'], 'gray']
        
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title(f"Match Outcome Probabilities ({num_simulations} simulations)")
        ax1.axis('equal')
        
        # Bar chart for most common scores
        top_scores = sorted_scores[:6]  # Show top 6 most common results
        score_labels = [x[0] for x in top_scores]
        score_probs = [x[1] for x in top_scores]
        
        ax2.bar(score_labels, score_probs, color='skyblue')
        ax2.set_title("Most Likely Scorelines")
        ax2.set_xlabel("Score (Home-Away)")
        ax2.set_ylabel("Probability (%)")
        
        # Add text for average goals
        avg_home_goals = home_goals_total / num_simulations
        avg_away_goals = away_goals_total / num_simulations
        
        fig.suptitle(f"{home_team} vs {away_team}\nAvg. Goals: {avg_home_goals:.2f} - {avg_away_goals:.2f}", fontsize=16)
        
        plt.tight_layout()
        return plt

    def create_interactive_dashboard(self):
        """Create an interactive dashboard with widgets"""
        # Create simulation widgets
        simulations_slider = widgets.IntSlider(
            value=1000,
            min=100,
            max=10000,
            step=100,
            description='Simulations:',
            style={'description_width': 'initial'}
        )
        
        run_button = widgets.Button(
            description='Run Simulations',
            button_style='primary',
            tooltip='Run Monte Carlo simulations'
        )
        
        # Create team selection widgets for match simulation
        team_dropdown1 = widgets.Dropdown(
            options=self.team_names,
            value=self.team_names[0],
            description='Home Team:',
            style={'description_width': 'initial'}
        )
        
        team_dropdown2 = widgets.Dropdown(
            options=self.team_names,
            value=self.team_names[1],
            description='Away Team:',
            style={'description_width': 'initial'}
        )
        
        simulate_match_button = widgets.Button


        # Add this to the bottom of the previous code file

def main():
    # Initialize the simulation
    simulation = PremierLeagueSimulation()

    # Number of Monte Carlo iterations to run
    num_simulations = input("Enter the number of Monte Carlo simulations to run: ")
    num_simulations = int(num_simulations)

    # Run the simulation
    simulation.run_monte_carlo(num_simulations)

    print("\n=== Premier League Simulation Results ===\n")

    # Show average table
    avg_table = simulation.show_average_table()
    print("\nAverage League Table (across all simulations):")
    display(avg_table)

    # Plot results
    print("\nGenerating visualization plots...\n")

    # Plot champion probabilities
#    plt.figure(figsize=(12, 6))
    simulation.plot_champion_probabilities()
    plt.show()

    # Plot position probabilities
 #   plt.figure(figsize=(14, 8))
    simulation.plot_position_probabilities()
    plt.show()

    # Plot expected points distribution
  #  plt.figure(figsize=(12, 8))
    simulation.plot_expected_points_distribution()
    plt.show()

    # Plot top-four probabilities
   # plt.figure(figsize=(12, 6))
    simulation.plot_top_four_probabilities()
    plt.show()

    # Interactive match simulation
    print("\n=== Interactive Match Simulation ===")
    team1 = "Man_City"
    team2 = "Liverpool"
    print(f"Simulating {team1} vs {team2}...")

    simulation.simulate_single_match_interactive(team1, team2)
    plt.show()

    print("\nSimulation complete! Try different teams or run more simulations to refine the results.")

# Execute the simulation if this file is run directly
if __name__ == "__main__":
    main()
