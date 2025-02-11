def calcDistance(point1, point2):
    # compute distance between two points."
    return sum((p1 - p2) ** 2 for p1, p2 in zip(point1, point2)) ** 0.5

def kmeans(data, k, max_iters=100):
    # randomly initialize k centroids from the data
    centroids = [data[i] for i in range(k)]
    
    for _ in range(max_iters):
        labels = []
        for point in data:
            distances = [calcDistance(point, centroid) for centroid in centroids]
            labels.append(distances.index(min(distances)))
        
        old_centroids = [c[:] for c in centroids]
        
        # update centroids by taking mean of assigned points
        new_centroids = [[0] * len(data[0]) for _ in range(k)]
        counts = [0] * k
        
        for point, label in zip(data, labels):
            counts[label] += 1
            for i in range(len(point)):
                new_centroids[label][i] += point[i]
        
        for i in range(k):
            if counts[i] > 0:
                centroids[i] = [x / counts[i] for x in new_centroids[i]]
        
        # check for convergence (if centroids don't move)
        if old_centroids == centroids:
            break
    
    return centroids, labels

# k-means usage
if __name__ == "__main__":
    # sample data
    data = [[(i % 5) - 2, (i % 7) - 3] for i in range(100)]  # 100 points in 2D
    centroids, labels = kmeans(data, k=3)
    
    print("Centroids:", centroids)
    print("\nFirst few labels:", labels[:10])
