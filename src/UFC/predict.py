import pandas as pd
import joblib  # Correct import for joblib

# Load the model
model = joblib.load('ufc_predictor_model.joblib')

# Define input data
data = {
    'BlueAvgTDLanded': [0],                      # Example value
    'AgeDif': [0],                               # Example value
    'BlueWinsByTKODoctorStoppage': [0],        # Example value
    'RedTotalRoundsFought': [0],                 # Example value
    'RWStrawweightRank': [0],                    # Example value
    'RedOdds': [0],                              # Example value
    'BlueAvgSigStrPct': [0],                     # Example value
    'RMatchWCRank': [0],                         # Example value
    'RKOOdds': [0],                              # Example value
    'LoseStreakDif': [0],                        # Example value
    'BlueWinsByKO': [0],                         # Example value
    'RedHeightCms': [0],                         # Example value
    'AvgTDDif': [0],                             # Example value
    'FinishRoundTime': [0],                      # Example value
    'BlueLongestWinStreak': [0],                 # Example value
    'BlueWinsByDecisionUnanimous': [0],          # Example value
    'RedCurrentLoseStreak': [0],                 # Example value
    'RSubOdds': [0],                             # Example value
    'BSubOdds': [0],                             # Example value
    'ReachDif': [0],                             # Example value
    'RedWeightLbs': [0],                         # Example value
    'BWFeatherweightRank': [0],                  # Example value
    'BWelterweightRank': [0],                     # Example value
    'BMiddleweightRank': [0],                     # Example value
    'RedAvgSigStrLanded': [0],                   # Example value
    'HeightDif': [0],                            # Example value
    'RWBantamweightRank': [0],                   # Example value
    'BWStrawweightRank': [0],                    # Example value
    'BFeatherweightRank': [0],                   # Example value
    'RedAvgTDPct': [0],                          # Example value
    'RWFeatherweightRank': [0],                  # Example value
    'TotalRoundDif': [0],                        # Example value
    'FinishRound': [0],                          # Example value
    'WeightClass': ['Middleweight'],              # Example value
    'LossDif': [0],                              # Example value
    'BlueWinsByDecisionMajority': [0],           # Example value
    'RHeavyweightRank': [0],                     # Example value
    'WinDif': [0],                               # Example value
    'BlueAvgSubAtt': [0],                        # Example value
    'BLightHeavyweightRank': [0],                # Example value
    'BlueAge': [30],                             # Example value
    'BlueDecOdds': [0],                          # Example value
    'NumberOfRounds': [3],                       # Example value
    'BlueExpectedValue': [0],                    # Example value
    'BHeavyweightRank': [0],                     # Example value
    'RWFlyweightRank': [0],                      # Example value
    'BKOOdds': [0],                              # Example value
    'RedAvgSubAtt': [0],                         # Example value
    'TitleBout': [0],                            # Example value
    'RMiddleweightRank': [0],                    # Example value
    'BetterRank': [0],                           # Example value
    'RedAvgTDLanded': [0],                       # Example value
    'BlueOdds': [0],                             # Example value
    'RedReachCms': [0],                          # Example value
    'BlueLosses': [0],                           # Example value
    'BlueHeightCms': [0],                        # Example value
    'RedDecOdds': [0],                           # Example value
    'BWFlyweightRank': [0],                      # Example value
    'BlueWinsBySubmission': [0],                 # Example value
    'BlueTotalRoundsFought': [0],                # Example value
    'BlueCurrentLoseStreak': [0],                # Example value
    'BlueReachCms': [0],                         # Example value
    'RedWinsByTKODoctorStoppage': [0],          # Example value
    'RBantamweightRank': [0],                    # Example value
    'RedCurrentWinStreak': [0],                  # Example value
    'RedWinsByDecisionSplit': [0],               # Example value
    'BBantamweightRank': [0],                    # Example value
    'BPFPRank': [0],                             # Example value
    'LongestWinStreakDif': [0],                  # Example value
    'BlueDraws': [0],                            # Example value
    'KODif': [0],                                # Example value
    'BMatchWCRank': [0],                         # Example value
    'BlueWinsByDecisionSplit': [0],              # Example value
    'RedWinsByDecisionUnanimous': [0],           # Example value
    'BLightweightRank': [0],                     # Example value
    'RFlyweightRank': [0],                       # Example value
    'WinStreakDif': [0],                         # Example value
    'RedExpectedValue': [0],                     # Example value
    'BlueWins': [0],                             # Example value
    'RLightHeavyweightRank': [0],                # Example value
    'BlueTotalTitleBouts': [0],                  # Example value
    'RedTotalTitleBouts': [0],                   # Example value
    'BFlyweightRank': [0],                        # Example value
    'RedWinsBySubmission': [0],                   # Example value
    'TotalFightTimeSecs': [0],                   # Example value
    'AvgSubAttDif': [0],                         # Example value
    'BWBantamweightRank': [0],                   # Example value
    'BlueStance': ['Orthodox'],                  # Example value
    'EmptyArena': [0],                           # Example value
    'BlueAvgTDPct': [0],                         # Example value
    'BlueAvgSigStrLanded': [0],                  # Example value
    'RedLongestWinStreak': [0],                  # Example value
    'SubDif': [0],                               # Example value
    'RedLosses': [0],                            # Example value
    'Gender': ['M'],                             # Example value
    'TotalTitleBoutDif': [0],                    # Example value
    'RedStance': ['Orthodox'],                   # Example value
    'RedWinsByKO': [0],                          # Example value
    'SigStrDif': [0],                            # Example value
    'BlueCurrentWinStreak': [0],                 # Example value
    'RLightweightRank': [0],                     # Example value
    'RPFPRank': [0],                             # Example value
    'RedAge': [31],                              # Example value
    'RedAvgSigStrPct': [0],                      # Example value
    'RedDraws': [0],                             # Example value
    'RWelterweightRank': [0],                    # Example value
    'RedWinsByDecisionMajority': [0],            # Example value
    'BlueWeightLbs': [0],                        # Example value
    'RedWins': [0],                              # Example value
    'RFeatherweightRank': [0]                    # Example value
}

# Create DataFrame
input_df = pd.DataFrame(data)

# Make prediction
try:
    prediction = model.predict(input_df)
    winner = "Red Fighter" if prediction == 1 else "Blue Fighter"
    print(f"The predicted winner is: {winner}")
except Exception as e:
    print(f"Error making prediction: {str(e)}")