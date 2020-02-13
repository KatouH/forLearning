#include"DataStruct.h"
#include<string>
#include<vector>
#include<unordered_map>
#include<algorithm>
#include<climits>
#include<stack>
using namespace std;


void myPrint(vector<int>& data);

bool isPalindrome(int x);//Accepted
int romanToInt(string s);//Accepted
ListNode* mergeTwoLists(ListNode* l1, ListNode* l2);//Accepted
void deleteNode(ListNode* node);//Accepted
int maxSubArray(vector<int>& nums);
int climbStairs(int n);
string longestCommonPrefix(vector<string>& strs);//Accepted
void merge(vector<int>& nums1, int m, vector<int>& nums2, int n);//Accepted
void mergeS1(vector<int>& nums1, int m, vector<int>& nums2, int n);//Other solution
int maxProfit(vector<int>& prices);//Ac r
int maxProfit2(vector<int>& prices);//Ac r
bool isPowerOfTwo(int n);//Ac 仅正数
string reverseWords(string s);//Ac r
bool isSymmetric(TreeNode* root);//Ac r
int rob(vector<int>& nums);//Ac dp r
bool isPalindrome(ListNode* head);//Ac r
int hammingDistance(int x, int y);//Ac 二进制 |或 &与 ^异或 ~取反 >>左移 <<右移
TreeNode* convertBST(TreeNode* root);

// 155.min-stack
class MinStack {
public:
	/** initialize your data structure here. */
	struct ListNode {
		int val;
		ListNode* next = NULL;
		ListNode(int v) :val(v) {};
	}*stackHead=NULL;

	MinStack() {
	}

	void push(int x) {
		ListNode* newNode = new ListNode(x);
		newNode->next = stackHead;
		stackHead = newNode;
	}

	void pop() {
		ListNode* popNode = stackHead;
		stackHead = stackHead->next;
		delete popNode;
	}

	int top() {
		return stackHead->val;
	}

	int getMin() {
		ListNode* currNode = stackHead;
		int min = INT_MAX;
		while (currNode != NULL) {
			if (currNode->val < min) {
				min = currNode->val;
			}
			currNode = currNode->next;
		}
		return min;
	}

	void printStack() {
		ListNode* currNode = stackHead;
		while (currNode != NULL) { 
			cout << currNode->val << endl;
			currNode = currNode->next;
		}
	}
};


int main() {
	TreeNode* root = new TreeNode(5);
	root->left = new TreeNode(2);
	root->right = new TreeNode(13);
	system("pause");
}

bool isPalindrome(int x) {
	string str = to_string(x);
	int r = 0, l = str.length();
	for (int r = 0, l = str.length() - 1; r < l; r++, l--) {
		if (str[r] != str[l])return false;
	}
	return true;
}

int romanToInt(string s) {
	int c[7] = { 0 };
	int reduction = 0;
	for (int i = 0; i < s.length(); i++) {
		switch (s[i]) {
		case 'I':
			if (i + 1 < s.length() && (s[i + 1] == 'V' || s[i + 1] == 'X'))reduction += 1;
			else c[0]++;
			break;
		case 'V':
			c[1]++;
			break;
		case 'X':
			if (i + 1 < s.length() && (s[i + 1] == 'L' || s[i + 1] == 'C'))reduction += 10;
			else c[2]++;
			break;
		case 'L':
			c[3]++;
			break;
		case 'C':
			if (i + 1 < s.length() && (s[i + 1] == 'D' || s[i + 1] == 'M'))reduction += 100;
			else c[4]++;
			break;
		case 'D':
			c[5]++;
			break;
		case 'M':
			c[6]++;
			break;
		}
	}
	return 1000 * c[6] + 500 * c[5] + 100 * c[4] + 50 * c[3] + 10 * c[2] + 5 * c[1] + c[0] - reduction;
}

void display(ListNode* result) {
	for (; result != NULL; result = result = result->next) {
		cout << result->val << " ";
	}
	cout << endl;
}

ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
	if (l1 == NULL)return l2;
	if (l2 == NULL)return l1;
	ListNode* remainL1 = l1;
	ListNode* currL1 = l1;
	ListNode* preL2 = NULL;
	ListNode* currL2 = l2;

	while (remainL1 != NULL) {
		currL1 = remainL1;
		remainL1 = remainL1->next;
		for (; currL2 != NULL; currL2 = currL2->next) {
			if (currL2->val > currL1->val) {
				currL1->next = currL2;
				if (preL2 == NULL) {
					l2 = currL1;
				}
				else {
					preL2->next = currL1;
				}
				break;
			}
			preL2 = currL2;
		}
		if (currL2 == NULL) {
			currL1->next = NULL;
			preL2->next = currL1;
		}
		preL2 = NULL;
		currL2 = l2;
	}
	return l2;
}



void deleteNode(ListNode* node) { //删除链表节点，将删除节点改为下一节点的副本，并删除下一个节点。
	ListNode* deleteNode = node->next;
	node->val = node->next->val;
	node->next = node->next->next;
	delete deleteNode;
}

bool hasCycle(ListNode *head) {//使用MAP存储节点，当节点数第二次出现时说明出现环
	unordered_map<ListNode*, int> map;
	for (ListNode* currNode = head; currNode != NULL; currNode = currNode->next) {
		if (map.count(currNode) > 0)return true;
		map[currNode] = currNode->val;
	}
	return false;
}

int maxSubArray(vector<int>& nums) {
	int maxSum = 0;
	for (int i = 0; i < nums.size(); i++) {
		int tmpNum = nums[i];
		int sum = nums[i];
		for (int j = i + 1; j < nums.size(); j++) {
			sum += nums[j];
			if (tmpNum < sum)tmpNum = sum;
		}
		if (i == 0) { maxSum = tmpNum; }
		if (tmpNum > maxSum)maxSum = tmpNum;
	}
	return maxSum;
}

int climbStairs(int n) {//dp问题
	int step0 = 1, step1 = 1, step = 0;
	for (int i = 2; i <= n; i++) {
		step = step0 + step1;
		step0 = step1;
		step1 = step;
	}
	return (n >= 2 ? step : (n == 1 ? step1 : 0));
}

int maxDepth(TreeNode* root) {
	int depth = 0;
	if (root == NULL)return depth;
	depth = max(maxDepth(root->left), maxDepth(root->right)) + 1;
	return depth;
}

ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {//https://leetcode-cn.com/problems/intersection-of-two-linked-lists/solution/xiang-jiao-lian-biao-by-leetcode/
	//hashmap  8790087
	unordered_map<ListNode*, int> List;
	for (auto cNode = headA; cNode != NULL; cNode = cNode->next) {
		List[cNode] = cNode->val;
	}
	for (auto cNode = headB; cNode != NULL; cNode = cNode->next) {
		if (List.count(cNode) > 0) {
			return cNode;
		}
	}
	return NULL;
}

//14.longest-common-prefix
string longestCommonPrefix(vector<string>& strs) {
	string commPrefix = "";
	if (strs.size() == 0) {
		return commPrefix;
	}
	commPrefix = strs[0];
	for (int i = 1; i < strs.size(); i++) {
		string temp = "";
		int minLen = min(commPrefix.size(), strs[i].size());
		for (int j = 0; j < minLen; j++) {
			if (commPrefix[j] == strs[i][j]) {
				temp += commPrefix[j];
			}
			else break; //不跳出循环会比较整个字符串而不是前缀
		}
		commPrefix = temp;
		if (commPrefix == "")
			return commPrefix;
	}
	return commPrefix;
}

//88.merge-sorted-array
void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
	if (m == 0) {
		nums1 = nums2;
		return;
	}
	if (n == 0) {
		return;
	}
	for (int i = m + n - 1; i > -1; i--) {
		if (m>0 && (nums1[m - 1] >= nums2[n - 1])) { 
			nums1[i] = nums1[m - 1]; 
			m--;
		}
		else {
			nums1[i] = nums2[n - 1];
			n--;
		}
		if (n == 0)break;
	}
}

void mergeS1(vector<int>& nums1, int m, vector<int>& nums2, int n) {
	nums1 = vector<int>(nums1.begin(), nums1.begin() + m);
	nums1.insert(nums1.end(), nums2.begin(), nums2.end());
	sort(nums1.begin(), nums1.end());
}

void myPrint(vector<int>& data) {
	for (auto c : data) {
		cout << c << " ";
	}
	cout << endl;
}

// 121.best-time-to-buy-and-sell-stock //iterator begin end 没有顺序大小关系
int maxProfit(vector<int>& prices) {
	int maxProfit = 0;
	if (prices.size() < 2)return maxProfit;
	for (auto s = prices.begin(); s != prices.end()-1; s++) {
		for (auto e = prices.end()-1; e != s; e--) {
			if (*s < *e) {
				if (maxProfit < (*e - *s)) {
					maxProfit = *e - *s;
				}
			}
		}
	}
	return maxProfit;
}
// 局部最优解->全局最优解 一次遍历
int maxProfitS1(vector<int>& prices) {
	int minPrice = INT_MAX;
	int maxProfit = 0;
	for (int i = 0; i < prices.size(); i++) {
		if (prices[i] < minPrice) {
			minPrice = prices[i];
		}
		else if(prices[i]-minPrice>maxProfit){
			maxProfit = prices[i] - minPrice;
		}
	}
	return maxProfit;
}

// 122.best-time-to-buy-and-sell-stock-ii //???
int maxProfit2(vector<int>& prices) {
	//7,1,5,3,6,4
	int currMin = 0;
	int sumProfit = 0;
	int i = 0;
	while (i < (int)prices.size()-1) {
		while (i<prices.size() - 1 && prices[i] >= prices[i + 1])i++;
		currMin = prices[i];
		while (i < prices.size() - 1 && prices[i] <= prices[i + 1])i++;
		sumProfit += (prices[i] - currMin);
	}
	return sumProfit;
}

// 169.majority-element
int majorityElement(vector<int>& nums) {
	unordered_map<int, int> cou;
	for (int i = 0; i < nums.size(); i++) {
		if (cou.count(nums[i])==0) {
			cou[nums[i]] = 1;
		}
		else {
			++cou[nums[i]];
		}
		if (cou[nums[i]] > nums.size() / 2)
			return nums[i];
	}
	return 0;
}

// 206.reverse-linked-list-iter 1->2->3->NULL 
ListNode* reverseList(ListNode* head) {
	ListNode* currNode = NULL;
	ListNode* tmpNode = head;

	while (tmpNode != NULL) {
		head = tmpNode;
		tmpNode = tmpNode->next;
		head->next = currNode;
		currNode = head;
	}
	return head;
}

// ??
ListNode* reverseListCurr(ListNode* head) {
	return NULL;
}

 //power-of-two
bool isPowerOfTwo(int n) {
	int i = 0;
	int j = 1;
	while (j > 0) {
		if (n == j)return true;
		j = j << 1;
		cout << j << endl;
	}
	return false;
}

// 235.lowest-common-ancestor-of-a-binary-search-tree 递归
TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
	TreeNode* commParent = root;
	if (p->val > root->val&&q->val > root->val) {
		commParent = lowestCommonAncestor(commParent->right, p, q);
	}
	else if (p->val < root->val&&q->val < root->val) {
		commParent = lowestCommonAncestor(commParent->left, p, q);
	}

	return commParent;
}
//二叉搜索数 left->val < root->val < right->val

// 292.
bool canWinNim(int n) {
	return !(n % 4 == 0);
}

// 557.
string reverseWords(string s) {
	string tmp = "";
	string news = "";
	for (int i = 0; i < s.size(); i++) {
		if (s[i] != ' ') {
			tmp += s[i];
		}
		else {
			reverse(tmp.begin(), tmp.end());
			news = news + tmp + " ";
			tmp = "";
		}
	}
	reverse(tmp.begin(), tmp.end());
	news += tmp;
	return news;
}


bool isMirror(const TreeNode* tree1,const TreeNode* tree2) {
	if (tree1 == NULL && tree2 == NULL)return true;
	if (tree1 == NULL || tree2 == NULL)return false;

	return (tree1->val == tree2->val) && isMirror(tree1->left, tree2->right) && isMirror(tree1->right, tree2->left);
}
// 101.symmetric-tree //用两个数镜像遍历
bool isSymmetric(TreeNode* root) {
	//中序遍历 
	//stack<TreeNode*> sroot;
	//TreeNode* currNode = root;
	//vector<int> middSeq;
	//while (currNode != NULL) {
	//	sroot.push(currNode);
	//	currNode = currNode->left;
	//}

	//while(!sroot.empty()) {
	//	currNode = sroot.top();
	//	sroot.pop();
	//	middSeq.push_back(currNode->val);
	//	if (currNode->right != NULL) {
	//		currNode = currNode->right;
	//		while (currNode != NULL) {
	//			sroot.push(currNode);
	//			currNode = currNode->left;
	//		}
	//	}
	//}
	
	return 	isMirror(root, root);
}

// 198.house-robber
int rob(vector<int>& nums) {
	int sumMax = 0;
	int k_1 = 0;
	int k_2 = 0;
	for (int i = 0; i < nums.size(); i++) {
		sumMax = max(k_1, k_2+nums[i]);
		k_2 = k_1;
		k_1 = sumMax;
	}
	return sumMax;
}

TreeNode* invertTree(TreeNode* root) {
	if (root == NULL)return root;
	stack<TreeNode*> sroot;
	TreeNode* currNode;
	sroot.push(root);
	while (!sroot.empty()) {
		currNode = sroot.top();
		sroot.pop();
		TreeNode* tmpNode = currNode->right;
		currNode->right = currNode->left;
		currNode->left = tmpNode;
		if (currNode->left != NULL) {
			sroot.push(currNode->left);
		}
		if (currNode->right != NULL) {
			sroot.push(currNode->right);
		}
	}
	return root;
}

// 其他解法：快慢指针 找到中间位置（慢指针走一步，快指针走两步）
// 反转指针比较
// 递归比较？？？
bool isPalindrome(ListNode* head) {
	if (head == NULL)return head;
	string list;
	for (auto curr = head; curr != NULL; curr = head->next) {
		cout << curr->val << " " << curr->val + '0' << endl;
		list += (curr->val + '0');
	}
	string prelist = list;
	reverse(list.begin(), list.end());
	return (prelist == list);
}

// 448. find-all-numbers-disappeared-in-an-array
vector<int> findDisappearedNumbers(vector<int>& nums) {

	vector<int> res;

	for (int i = 0; i < nums.size(); i++) {
		while (nums[i] != (i + 1)) {
			if (nums[nums[i] - 1] != nums[i]) {
				int tmp = nums[nums[i] - 1];
				nums[nums[i] - 1] = nums[i];
				nums[i] = tmp;
			}
			else {
				break;
			}
		}
	}

	for (int i = 0; i < nums.size(); i++) {
		if (nums[i] != (i + 1)) {
			res.push_back(i + 1);
		}
	}

	return res;
}

// 461.hamming-distance
int hammingDistance(int x, int y) {
	int dis = 0;
	int res = x ^ y;
	while (res != 0) {
		if (res & 1 == 1)dis++;
		res = res >> 1;
	}
	return dis;
}

// 538.convert-bst-to-greater-tree 中序遍历思想
TreeNode* convertBST(TreeNode* root) {
	stack<TreeNode*> sroot;
	TreeNode* currNode=root;
	vector<TreeNode*> midSeq;
	vector<int> middSeq;

	while (currNode != NULL) {
		sroot.push(currNode);
		currNode = currNode->left;
	}

	while (!sroot.empty()) {
		currNode = sroot.top();
		sroot.pop();
		for (auto &a : midSeq) {
			a->val += currNode->val;
			cout << a->val << " ";
		}
		cout << endl;
		midSeq.push_back(currNode);
		middSeq.push_back(currNode->val);
		if (currNode->right != NULL) {
			currNode = currNode->right;
			while (currNode != NULL) {
				sroot.push(currNode);
				currNode = currNode->left;
			}
		}
	}
	return root;
}
// 从右向左，每个节点将从上一位置的累加数加到自己的节点中,累加数为全局
int addSum = 0;
TreeNode* convertBSTS1(TreeNode* root) {
	if (root == NULL)return root;
	convertBSTS1(root->right);
	root->val += addSum;
	addSum = root->val;
	convertBSTS1(root->left);
	return root;
}