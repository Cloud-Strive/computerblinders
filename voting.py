import voterSystem
from referenceVotingSystem import VotingSystem

with open("valid_student_numbers.txt", "r") as file:
    valid_student_numbers = file.read().splitlines()
for student_number in valid_student_numbers:
    VotingSystem.register_voter(student_number)
while True:
    voter_id = input("Enter your voter ID (type 'done' to finish): ")
    if voter_id.lower() == 'done':
        break
    VotingSystem.register_voter(voter_id)

# Candidate nomination
voter_id = input("Enter your voter ID to nominate candidates: ")
candidates = []
for i in range(3):
    candidate = input(f"Enter candidate {i+1}: ")
    candidates.append(candidate)
VotingSystem.nominate_candidates(voter_id, candidates)

# Voting process
'''voter_id = input("Enter your voter ID to vote: ")
candidate = input("Enter the candidate you want to vote for: ")
VotingSystem.vote(voter_id, candidate)'''
while True:
    voter_id = input("Enter your voter ID to vote (type 'done' to finish): ")
    if voter_id.lower() == 'done':
        break
    candidate = input("Enter the candidate you want to vote for: ")
    VotingSystem.vote(voter_id, candidate)
# Count votes and visualize results
results = VotingSystem.count_votes()
VotingSystem.plottingResults(results)

"""
NumberOfVoters = int(input("Enter the number of voters : "))
lyst = []
for i in range (0, NumberOfVoters):
    t = input("Enter a valid student number: ")
    lyst.append(t)
print(lyst)
VotingSystem.nominate_candidates("12345", lyst, ["Cloud " ,"Senzo ", "Bennett"])

voterSystem.vote(lyst[1],"Cloud Strive")
results = voterSystem.count_votes()
voterSystem.visualize_results(results)
"""