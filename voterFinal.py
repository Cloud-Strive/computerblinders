import matplotlib.pyplot as plt
from linkers import LinkedList

class VotingSystem:
    def __init__(self):
        self.voters_list = LinkedList()
        self.candidates = {}
        self.votes = {}


#registering voters and checking for registered voters

    def register_voter(self, voter_id):
        if not self.voters_list.search(voter_id):
            self.voters_list.append(voter_id)
            return True
        else:
            print("Voter already registered.")
            return False

#Function to nominate three candidates
    def nominate_candidates(self, voter_id, candidates):
        if len(candidates) != 3:
            print("Please nominate exactly three candidates.")
            return False
        if not self.voters_list.search(voter_id):
            print("Voter is not registered.")
            return False
        self.candidates[voter_id] = candidates
        return True

# checking if voter id is registered for voting and allow valid ids to vote
    def vote(self, voter_id, candidate):
        if not self.voters_list.search(voter_id):
            print("Voter is not registered.")
            return False
        if voter_id in self.votes:
            print("Voter has already voted.")
            return False
        if candidate not in self.candidates[voter_id]:
            print("Candidate not nominated by the voter.")
            return False
        self.votes[voter_id] = candidate
        return True

# addding votes to the condidates' count dictionary and incrementing
    def count_votes(self):
        results = {}
        for candidate in self.candidates.values():
            for c in candidate:
                results[c] = 0
        for vote in self.votes.values():
            results[vote] += 1
        return results

# displaying result in a chart
    def plottingResults(self, results):
        labels = results.keys()
        values = results.values()
        plt.bar(labels, values)
        plt.xlabel('Candidates')
        plt.ylabel('Votes')
        plt.title('Election Results')
        plt.show()
