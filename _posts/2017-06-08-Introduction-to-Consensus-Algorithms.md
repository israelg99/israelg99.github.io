---
layout: post
title: Introduction to Consensus Algorithms
category: database
---
When bored and talented engineers get all their real-life problems figured out and are searching for new exciting problems to solve, they all stumble upon this interesting field of computer science called - distributed computing.

Distributed computing is where a set of computers interact with each other to achieve a common goal.

#### But not so quickly...
Computer programs, similarily to geeks, have difficulties sometimes agreeing with each other, for example they may disagree on some data value, this is especially problematic in systems where performance is critical.  

Fortunately, there are various algorithms available to solve this issue quite effectivly, those algorithms are called consensus algorithms, Raft and Paxos are good examples.

Paxos and Raft are similar algorithms, but Raft is easier to understand and generally a better algorithm, let's jump right in.

### Raft
![]({{ site.baseurl }}/images/2017-06-07-Introduction-to-Consensus-Algorithms/raft-logo.png){:height="240x" width="240px"}
Raft started as a goal to take the unecessery complex Paxos algorithm and strip away all of its shittiness.

This resulted in Raft being simpler and more understandable algorithm than Paxos yet being safe and even have additional features.

Raft solves the consensus problem by having a leader node in a cluster, all changes go through that leader and the leader sends updates to the rest of the cluster, if the leader is gone missing and stops sending updates, the cluster starts a new leader election, and a new leader is elected.

#### Let's get more technical
Raft achieves consensus via an elected leader.  
A node in a raft cluster is either a leader, a candidate, or a follower.  
The leader regularly informs the followers of its existence by sending a heartbeat messages.
Each follower has a timeout (around 200 ms) in which it expects the heartbeat from the leader.  
The timeout is reset on receiving the heartbeat.  
If no heartbeat is received the follower changes its status to candidate and starts a new leader election.

#### How leader election works?
A leader election is started by a candidate node.  
A node becomes a candidate if it receives no heartbeat from the leader within the timeout.  
It starts the election by requesting votes from all other nodes.  
The other nodes will vote for the first candidate that sends them a vote request.  
A node will only vote once per term.  
If the candidate receives a message from a leader with a term number equal to or larger than the current term, then its election is defeated and the candidate changes into a follower.  
If a candidate receives a majority of votes, then it becomes the new leader.  
If neither happens, e.g., because of a split vote, then a new leader election is started after a timeout.

The timeout values of each node should be spread out within a reasonable interval. This should reduce the chance of a split vote because nodes won't become candidates at the same time.

#### How updates work?
This process is called Log Replication in Raft.  
Whenever the leader receives new changes from the client, the changes are not yet commited, the leader first sends an update to all of its followers and waits for their confirmation, once the leader receives confirmation from the majority of its followers the changes are commited.

### Conclusion
I hope this gives you 