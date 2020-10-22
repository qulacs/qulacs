#include <vector>
#include <utility>
class UnionFind{
private:
	std::vector<int> Parent;
public:
	UnionFind(int N) {
		Parent = std::vector<int>(N, -1);
	}
	int root(int A) {
		if (Parent[A] < 0) {
			return A;
		}
		return Parent[A] = root(Parent[A]);
	}

	bool same(int A, int B) {
		return root(A) == root(B);
	}

	int size(int A) {
		return -Parent[root(A)];
	}

	bool connect(int A, int B) {
		A = root(A);
		B = root(B);
		if (A == B) {
			return false;
		}
		if (size(A) < size(B)) {
			std::swap(A, B);
		}

		Parent[A] += Parent[B];
		Parent[B] = A;

		return true;
	}

};