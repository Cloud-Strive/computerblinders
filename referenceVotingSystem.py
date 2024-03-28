from voterFinal import VotingSystem
import matplotlib.pyplot as plt

voting_system = VotingSystem()

# Voter registration
'''with open("valid_student_numbers.txt", "r") as file:
    valid_student_numbers = file.read().splitlines()
for student_number in valid_student_numbers:
    voting_system.register_voter(student_number)
'''
# loop to get voter ids until finished with 'done'

while True:
    voter_id = input("Enter your voter ID (type 'done' to finish): ")
    if voter_id.lower() == 'done':
        break
    voting_system.register_voter(voter_id)

# Candidate nomination loop
voter_id = input("Enter your voter ID to nominate candidates: ")
candidates = []

for i in range(3):
    candidate = input(f"Enter candidate {i+1}: ")
    candidates.append(candidate)
voting_system.nominate_candidates(voter_id, candidates)
t = []
s = []
# Voting process
while True:
    voter_id = input("Enter your voter ID to vote (type 'done' to finish): ")
    #s.append(voter_id)
    if voter_id.lower() == 'done':
        break
    candidate = input("Enter the candidate you want to vote for: ")
    #t.append(candidate)


m = 0
n = 0

for number in s:
    m += int(number)
    try:
            n += int(number)
    except ValueError as e:
        print(f"Error: {e}")
for numbers in t:
    n += int(numbers)
    try:
        n += int(numbers)
    except ValueError as e:
        print(f"Error: {e}")
voting_system.vote(m, n)
print(m)
print(n)

def plottingResults(self, results):
    labels = results.keys()
    values = results.values()
    plt.bar(labels, values)
    plt.xlabel('Candidates')
    plt.ylabel('Votes')
    plt.title('Election Results')
    plt.show()

'''voter_id = input("Enter your voter ID to vote: ")
candidate = input("Enter the candidate you want to vote for: ")
voting_system.vote(voter_id, candidate)'''

# Count votes and visualize results
results = voting_system.count_votes()
voting_system.plottingResults(results)