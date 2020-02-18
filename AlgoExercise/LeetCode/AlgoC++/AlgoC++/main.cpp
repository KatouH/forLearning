#include"DataStruct.h"
#include<string>
#include<vector>
#include<unordered_map>
#include<algorithm>
#include<climits>
#include<stack>
#include<queue>
#include<set>
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
string countAndSay(int n);
int mySqrt(int x);
TreeNode* sortedArrayToBST(vector<int>& nums);
vector<vector<int>> generate(int numRows);
bool isPalindrome(string s);
bool isPalindromeS1(string s);
int titleToNumber(string s);
uint32_t reverseBits(uint32_t n);
bool isHappy(int n);
int missingNumber(vector<int>& nums);
int lengthOfLastWord(string s);
string addBinary(string a, string b);
ListNode* deleteDuplicates(ListNode* head);
vector<vector<int>> levelOrderBottom(TreeNode* root);
bool isBalanced(TreeNode* root);
int addDigits(int num);
string getHint(string secret, string guess);
string reverseVowels(string s);
int longestPalindrome(string s);

template<typename T>
void pVector(const vector<T>& vec) {
	for (auto c : vec) {
		cout << c << " ";
	}
	cout << endl;
}

// 一个队列 每次push调换顺序将队列前部至于后部
// 两个队列 一个存储栈顶元素；两个队列交换元素
class MyStack {
public:
	/** Initialize your data structure here. */
	queue<int> q1;
	MyStack() {
	}

	/** Push element x onto stack. */
	void push(int x) {
		q1.push(x);
		for (int i = q1.size(); i > 1; i--) {
			q1.push(q1.front());
			q1.pop();
		}
	}

	/** Removes the element on top of the stack and returns that element. */
	int pop() {
		int popEl = q1.front();
		q1.pop();
		return popEl;
	}

	/** Get the top element. */
	int top() {
		return q1.front();
	}

	/** Returns whether the stack is empty. */
	bool empty() {
		return q1.empty();
	}
};

class MyQueue {
public:
	/** Initialize your data structure here. */
	stack<int> s1;
	stack<int> s2;
	MyQueue() {

	}

	/** Push element x to the back of queue. */
	void push(int x) {
		s1.push(x);
	}

	/** Removes the element from in front of queue and returns that element. */
	int pop() {
		if (s2.empty()) {
			while (!s1.empty()) {
				s2.push(s1.top());
				s1.pop();
			}
		}
		int popEl = s2.top();
		s2.pop();
		return popEl;
	}

	/** Get the front element. */
	int peek() {
		if (s2.empty()) {
			while (!s1.empty()) {
				s2.push(s1.top());
				s1.pop();
			}
		}
		return s2.top();
	}

	/** Returns whether the queue is empty. */
	bool empty() {
		return s1.empty() && s2.empty();
	}
};

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

//****************************  Main **************************************
int main() {
	TreeNode* root = new TreeNode(5);
	root->left = new TreeNode(2);
	root->left->left = new TreeNode(3);
	root->right = new TreeNode(13);
	vector<int> test{ 3,0,1 };
	cout << longestPalindrome("civilwartestingwhetherthatnaptionoranynartionsoconceivedandsodedicatedcanlongendureWeareqmetonagreatbattlefiemldoftzhatwarWehavecometodedicpateaportionofthatfieldasafinalrestingplaceforthosewhoheregavetheirlivesthatthatnationmightliveItisaltogetherfangandproperthatweshoulddothisButinalargersensewecannotdedicatewecannotconsecratewecannothallowthisgroundThebravelmenlivinganddeadwhostruggledherehaveconsecrateditfaraboveourpoorponwertoaddordetractTgheworldadswfilllittlenotlenorlongrememberwhatwesayherebutitcanneverforgetwhattheydidhereItisforusthelivingrathertobededicatedheretotheulnfinishedworkwhichtheywhofoughtherehavethusfarsonoblyadvancedItisratherforustobeherededicatedtothegreattdafskremainingbeforeusthatfromthesehonoreddeadwetakeincreaseddevotiontothatcauseforwhichtheygavethelastpfullmeasureofdevotionthatweherehighlyresolvethatthesedeadshallnothavediedinvainthatthisnationunsderGodshallhaveanewbirthoffreedomandthatgovernmentofthepeoplebythepeopleforthepeopleshallnotperishfromtheearth") << endl;
	//cout<< <<endl;

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


// 581.shortest-unsorted-continuous-subarray
int findUnsortedSubarray(vector<int>& nums) {
	vector<int> preNums = nums;
	int len = 0;
	sort(nums.begin(), nums.end());
	int head(0), footer(nums.size() - 1);
	for (; head < nums.size(); head++) {
		if (nums[head] != preNums[head])break;
	}
	for (; footer > -1; footer--){
		if (nums[footer] != preNums[footer])break;
	}
	len = footer - head + 1;
	return (len>0?len:0);
}


// 617.merge-two-binary-trees
TreeNode* mergeTrees(TreeNode* t1, TreeNode* t2) {
	if (t1&&t2) {
		t1->val += t2->val;
	}
	else if (t1||t2) {
		return (t1 ? t1 : t2);
	}
	if (t1->left&&t2->left) {
		mergeTrees(t1->left, t2->left);
	}
	else if (t1->left == NULL && t2->left != NULL) {
		t1->left = t2->left;
	}
	if (t1->right&&t2->right) {
		mergeTrees(t1->right, t2->right);
	}
	else if (t1->right == NULL && t2->right != NULL) {
		t1->right = t2->right;
	}
	return t1;

}


// 38.count-and-say
string countAndSay(int n) {
	if (n == 1) {
		return "1";
	}
	string preSeq = countAndSay(n - 1);
	string currSeq = "";
	for (int i = 0; i < preSeq.size(); i++) {
		int count = 0;
		for (int j = i; j < preSeq.size(); j++) {
			if (preSeq[i] != preSeq[j]) { 
				i = j - 1;
				break; 
			}
			count++;
			i = j;
		}
		currSeq += to_string(count) + preSeq[i];
	}
	return currSeq;
}

int mySqrt(int x) {
	int i = 0;
	for (; i <= x / 2; i++) {
		if (i*i <= x && (i + 1)*(i + 1) > x)break;
		cout << i << endl;
	}
	return i;
}
 
TreeNode* sortedArrayToBST(vector<int>& nums) {
	int mid = nums.size() / 2;
	TreeNode* newNode = new TreeNode(nums[mid]);
	if (nums.size() == 1) {
		return newNode;
	}
	vector<int> left = vector<int>(nums.begin(), nums.begin() + mid);
	newNode->left = sortedArrayToBST(left);
	if (mid + 1 < nums.size()) {
		vector<int> right =  vector<int>(nums.begin() + mid + 1, nums.end());
		newNode->right = sortedArrayToBST(right);
	}
	return newNode;
}

vector<vector<int>> generate(int numRows) {
	vector<vector<int>> pascals{ {1},{1,1} };
	if (numRows <= 2)return vector<vector<int>>(pascals.begin(), pascals.begin() + numRows);
	for (int i = 2; i < numRows; i++) {
		vector<int> row{ 1 };
		for (int j = 1; j < i; j++) {
			row.push_back(pascals[i - 1][j - 1] + pascals[i - 1][j]);
		}
		row.push_back(1);
		pVector(row);
		pascals.push_back(row);
	}
	return pascals;
}

bool isPalindrome(string s) {
	if (s == "")return true;
	vector<char> tmp;
	for (auto c : s) {
		if ((c >= 'a'&&c <= 'z')||(c>='0'&&c<='9'))tmp.push_back(c);
		else if (c >= 'A'&&c <= 'Z')tmp.push_back(c - 'A' + 'a');
	}
	vector<char> preTmp = tmp;
	reverse(tmp.begin(), tmp.end());
	for (int i = 0; i < tmp.size(); i++) {
		if (preTmp[i] != tmp[i])return false;
	}
	return true;
}

// 双指针
bool isPalindromeS1(string s) {
	int pheader = 0;
	int pfooter = s.size()-1;
	while (pheader != s.size()) {
		if (!isalnum(s[pheader])) { pheader++;  continue; };
		if (!isalnum(s[pfooter])) { pfooter--;  continue;};
		if (tolower(s[pheader]) == tolower(s[pfooter])) {
			pheader++;
			pfooter--;
		}
		else {
			return false;
		}
	}
	return true;
}

int titleToNumber(string s) {
	int num=0;
	int count = 1;
	for (int i = s.size()-1; i > -1;i--) {
		cout << num << " " << count << endl;
		cout << s[i] - 'A' + 1 << endl;
		num += ((s[i] - 'A' + 1)*count);
		count *= 26;
	}
	return num;
}

int firstUniqChar(string s) {
	if (s == "")return -1;
	unordered_map<char, int> tmp;
	for (int i = 0; i < s.size(); i++) {
		if (tmp.count(s[i]) == 0)tmp[s[i]]=1;
		else {
			tmp[s[i]]++;
		}
	}
	for (int i = 0; i < s.size(); i++) {
		if (tmp[s[i]] == 1)return s[i];
	}
	return -1;
}

// (1&n)&res 获取当前位，现处理当前位再一位 循环31一位，移位32，处理最后一位
uint32_t reverseBits(uint32_t n) {
	uint32_t res=0;
	for (int i = 0; i < 31; i++) {
		res = (1 & n) | res;
		n = n >> 1;
		res = res << 1;
	}
	res = (1 & n) | res;
	return res;
}

int hammingWeight(uint32_t n) {
	int count = 0;
	for (int i = 0; i < 31; i++) {
		if (n & 1) {
			count++;
		}
		n >> 1;
	}
	if (n & 1) {
		count++;
	}
	return count;
}


int caResult(int n) {
	int tmp = n;
	int res=0;
	while (tmp != 0) {
		res += ((tmp % 10) * (tmp % 10));
		tmp = tmp / 10;
	}
	return res;
}
// 快慢指针可以判断是否存在循坏， 当慢指针走完一个周期，快指针正好走完两个周期（慢指针一步，快指针两部）
bool isHappy(int n) {
	
	long fast = n;
	long slow = n;
	int tmp = n;

	do {
		slow = caResult(slow);
		for (int i = 0; i < 2; i++) {
			fast = caResult(fast);
			if (fast == 1)return true;
		}
		cout << slow << " "<<fast << endl;

	} while (fast!=slow);
	return false;
}

//筛法
int countPrimes(int n) {
	int* arrPrimes = new int[n];
	int count = 0;
	for (int i = 0; i < n; i++) {
		arrPrimes[i] = i;
	}
	for (int i = 0; i <= (int)sqrt(n); i++) {
		if (i != 0 && i != 1) {
			for (int j = 2; j*i < n; j++) {
				if(arrPrimes[j*i]!=0)arrPrimes[j*i] = 0;
			}
		}
	}
	for (int i = 0; i < n; i++) {
		if (i != 0 && i != 1)count++;
	}

	return count;
}

int missingNumber(vector<int>& nums) {
	vector<int> nums_c( nums.size()+1,-1);
	for (int i = 0; i < nums.size(); i++) {
		nums_c[nums[i]] = nums[i];
	}
	for (int i = 0; i < nums_c.size(); i++) {
		if (nums_c[i] == -1)return i;
	}
	return 0;
}

bool isPowerOfThree(int n) {
	int tmp = n;
	if (n == 1)return true;
	while (tmp != 3) {
		if (tmp % 3 != 0)return false;
		tmp = tmp / 3;
	}
	return true;
}

//位运算实现加法器
int getSum(int a, int b) {
	return a + b;//？
}

vector<string> fizzBuzz(int n) {
	vector<string> str;
	for (int i = 1; i <= n; i++) {
		if (i % 15 == 0)str.push_back("FizzBuzz");
		else if (i % 3 == 0)str.push_back("Fizz");
		else if (i % 5 == 0)str.push_back("Buzz");
		else {
			str.push_back(to_string(i));
		}
	}
	return str;
}

// remove algorithm头文件中 删除vector元素 删除的元素被替换为默认值，并移动至尾部，并返回“删除”后新的end() 不改变size  
// erase vector 成员函数 删除指定位置元素 并返回下一个迭代器，改变size
// remove erase 联合使用删除所有指定元素 nums.erase(remove(begin,end,deleteEle),end);
int removeElement(vector<int>& nums, int val) {
	nums.erase(remove(nums.begin(), nums.end(), val), nums.end());
	return nums.size();
}

int searchInsert(vector<int>& nums, int target) {
	for (int i = 0; i < nums.size(); i++) {
		if (target == nums[i])return i;
		else if (target > nums[i])return i;
	}
	return 0;
}

int lengthOfLastWord(string s) {
	int len = 0;
	for (int i = s.size() - 1; i > -1; i--) {
		if (isalpha(s[i])) { len++; }
		else if (s[i] == ' '&&len != 0)break;
	}
	return len;
}

string addBinary(string a, string b) {
	if (a.size() == 0 || b.size() == 0)return (a.size() == 0 ? b : a);
	char c = '0';
	char bitRes = '0';
	string res = "";
	int d = a.size() - b.size();
	if (d > 0)b = string(d, '0') + b;
	else if (d < 0)a = string(-d, '0') + a;
	auto iter_a = a.end();
	auto iter_b = b.end();
	while (iter_a != a.begin() && iter_b != b.begin()) {
		iter_a--;
		iter_b--;
		bitRes = (*iter_a - '0') + (*iter_b - '0') + (c - '0')+'0';
		if (bitRes == '2') { c = '1'; bitRes = '0'; }
		else if (bitRes == '3') { c = '1'; bitRes = '1'; }
		else { c = '0'; }
		res = bitRes + res;
	}

	if (c != '0')res = c + res;

	return res;
}

// 1 1 2  双指针？
ListNode* deleteDuplicates(ListNode* head) {
	for (auto currNode = head; currNode != NULL;) {
		if (currNode->next != NULL) {
			if (currNode->val == currNode->next->val) {
				auto tmpNode = currNode->next;
				currNode->next = currNode->next->next;
				delete tmpNode;
			}
			else {
				currNode = currNode->next;
			}
		}
		else {
			currNode = currNode->next;
		}
	}
	return head;
}


//递归 循环？
bool isSameTree(TreeNode* p, TreeNode* q) {
	if (p == NULL && q == NULL)return true;
	else if (p == NULL || q == NULL)return false;
	
	return (p->val == q->val) && isSameTree(p->right, q->right) && isSameTree(p->left, q->left);
}

vector<vector<int>> levelOrderBottom(TreeNode* root) {
	if (root == NULL)return vector<vector<int>>();
	queue<TreeNode*> floorNode;
	TreeNode* currNode;
	vector<int> currFloor;
	vector<vector<int>> resFloor;
	vector<TreeNode*> currTFloor;

	floorNode.push(root);
	while (!floorNode.empty()) {
		while (!floorNode.empty()) {
			currNode = floorNode.front();
			floorNode.pop();
			currFloor.push_back(currNode->val);
			currTFloor.push_back(currNode);
			pVector(currFloor);
		}
		for (auto c : currTFloor) {
			if (c->left != NULL)floorNode.push(c->left);
			if (c->right != NULL)floorNode.push(c->right);
		}
		resFloor.push_back(currFloor);
		currTFloor.clear();
		currFloor.clear();
	}
	reverse(resFloor.begin(), resFloor.end());
	return resFloor;
}

// ++++++++++++++++++++++++++++++++++++++++++++++++递归获得深度
int getDepth(TreeNode* root) {
	if (root == NULL)return 0;
	int depth = max(getDepth(root->left), getDepth(root->right)) + 1;
	return depth;
}

bool isBalanced(TreeNode* root) {
	if (root == NULL)return true;
	return (abs(getDepth(root->left) - getDepth(root->right))<=1)&&isBalanced(root->right)&&isBalanced(root->left);
}


// 深度递归，注意处理仅在一个方向延伸的树！
int minDepth(TreeNode* root) {
	if (root == NULL)return 0;
	int minDep = 0;
	int leMin = minDepth(root->left);
	int riMin = minDepth(root->right);
	if(leMin!=0&&riMin!=0)minDep = min(leMin, riMin) + 1;
	else if (leMin != 0 || riMin != 0)minDep = (leMin == 0 ? riMin : leMin);
	else minDep = 1;
	return minDep;
}

//1         0
//1 1	    1
//1 2 1     2
//1 3 3 1   3->n
//只需要计算 [i-1,1]位置的内容
//从后往前 中间有该项等于上一项加上本项
//+++++++++++++++++++++++important
vector<int> getRow(int rowIndex) {
	vector<int> row;
	for (int i = 0; i <= rowIndex; i++) {
		row.push_back(1);
		for (int j = i - 1; j > 0; j--) {
			row[j] += row[j - 1];
		}
	}
	return row;
}

vector<int> twoSum(vector<int>& numbers, int target) {
	unordered_map<int, vector<int>> record;
	vector<int> res;
	for (int i = 0; i < numbers.size(); i++) {
		record[numbers[i]].push_back(i + 1);
	}
	for (int i = 0; i < numbers.size(); i++) {
		int tmp = target - numbers[i];
			if (record.find(tmp) != record.end()) {
				if (record[tmp][0] != i + 1) {
					res.push_back(min(record[tmp][0], i + 1));
					res.push_back(max(record[tmp][0], i + 1));
				}
				else {
					return (record[tmp].size() != 1 ? record[tmp] : vector<int>());
				}
		}
	}
	return res;
}

// 将1-26的数转化为0-25的26进制数 每除26减一
// x*26^n - x*26^0 其中 x(n->0)就表示的是各个位的位数(0-25)
string convertToTitle(int n) {
	int i(0), j(n);
	string res = "";
	while (j != 0) {
		j--;
		i = j % 26;
		j = j / 26;
		if(i!=0)res = char(i + 'A') + res;
	}
	return res;
}

// 数据库左连接？？？
//Select FirstName, LastName, City, State from Person left join Address on Person.PersonId = Address.PersonId


vector<int> toNumVec(string str) {
	int i = 0;
	unordered_map<char, int> map;
	vector<int> numVec;
	for (auto c : str) {
		if (map.find(c) == map.end())map[c] = i++;
	}
	for (auto c : str) {
		numVec.push_back(map[c]);
	}
	return numVec;
}

// 转换成数字串，每个数字表示当前字符是字符串中第几个新出现的字符
bool isIsomorphic(string s, string t) {
	vector<int> vecs = toNumVec(s);
	vector<int> vect = toNumVec(t);
	for (int i = 0; i < vecs.size(); i++) {
		if (vecs[i] != vect[i])return false;
	}
	return true;
}

bool containsNearbyDuplicate(vector<int>& nums, int k) {
	unordered_map<int, vector<int>> map;
	for (int i = 0; i < nums.size(); i++) {
		map[nums[i]].push_back(i);
	}
	for (int i = 0; i < nums.size(); i++) {
		if (map[nums[i]].size() >= 2) {
			for (int j = map[nums[i]].size() - 1; j > 0; j--) {
				if (map[nums[i]][j] - map[nums[i]][j - 1] <= k)return true;
			}
		}
	}
	return false;
}


// 递归  栈:每出现分支push进入容器
vector<string> pathVec;

void currTree(TreeNode* root, string path) {
	if (root->left == NULL && root->right == NULL) {
		path = to_string(root->val) + path;
		pathVec.push_back(path);
		return;
	}
	else {
		path = path + "->" + to_string(root->val);
		if (root->left == NULL)currTree(root->right, path);
		else if (root->right == NULL)currTree(root->left, path);
		else {
			currTree(root->left, path);
			currTree(root->right, path);
		}
	}
}

vector<string> binaryTreePaths(TreeNode* root) {
	if (root = NULL)return vector<string>();
	currTree(root, "");
}


//不用循环递归 O(1)??
int addDigits(int num) {
	int res(0), tmp(num);
	while (tmp >= 10) {
		while (tmp != 0) {
			res += (tmp % 10);
			tmp = tmp / 10;
			cout << tmp << " " << res << endl;
			system("pause");
		}
		tmp = res;
		res = 0;
	}
	return tmp;
}

bool isUgly(int num) {
	if (num == 0)return false;
	if (num == 1)return true;
	return (num % 2 == 0 ? isUgly(num / 2) : false) || (num % 3 == 0 ? isUgly(num / 3) : false) || (num % 5 == 0 ? isUgly(num / 5) : false);
}

//bool isBadVersion(int version);
//
//int firstBadVersion(int n) {
//	if (n == 1)return 1;
//	int midSearch = n / 2;
//	while (!isBadVersion(midSearch)&&midSearch!=n) {
//		midSearch = (midSearch + n) / 2;
//	}
//	int i = midSearch;
//	for (; isBadVersion(i) && isBadVersion(i - 1); i--);
//	return i;
//}

string genPattern(string pattern) {
	string gen = "";
	int count=0;
	unordered_map<char, char> map;
	for (auto c : pattern) {
		if (map.find(c) == map.end()) {
			gen += to_string(count);
			map[c] = (count++)+'0';
		}else{
			gen += map[c];
		}
	}
	return gen;
}

string genStr(string str) {
	string gen = "";
	int count = 0;
	unordered_map<string, char> map;
	string tmp = "";
	for (auto iter = str.begin(); iter != str.end()+1; iter++) {
		if (*iter != ' ' && iter != str.end()) {
			tmp += *iter;
		}
		else {
			if (map.find(tmp) == map.end()) {
				gen += to_string(count);
				map[tmp] = (count++)+'0';
			}
			else {
				gen += map[tmp];
			}
			tmp = "";
		}
	}
	return gen;
}

bool wordPattern(string pattern, string str) {
	return genPattern(pattern) == genStr(str);
}


//先找到全相等的元素 替换掉以避免干扰 再找位置不同的元素 找到后替换以免干扰
string getHint(string secret, string guess) {
	int bulls = 0;
	int cows = 0;
	for (int i = 0; i < secret.size(); i++) {
		if (secret[i] == guess[i]) { 
			bulls++;
			secret[i] = ' ';
			guess[i] = ' ';
		}
	}
	cout << secret << " " << guess << endl;
	for (int i = 0; i < secret.size(); i++) {
		if (secret[i] != ' '&&guess.find(secret[i]) != string::npos) {
			cows++;
			guess[guess.find(secret[i])] = ' ';
		}
	}
	return to_string(bulls) + "A" + to_string(cows) + "B";
}

vector<int> nums;
int sumRange(int i, int j) {
	int sum = 0;
	for (int n = i; n <= j; n++) {
		sum += nums[n];
	}
	return sum;
}

bool isPowerOfFour(int num) {
	unsigned powerFour = 1;
	while (powerFour != 0) {
		if (powerFour == powerFour | num)return true;
		else powerFour << 2;
	}
	return false;
}

vector<int> intersection(vector<int>& nums1, vector<int>& nums2) {
	unordered_map<int, int> map;
	vector<int> set;
	for (int i = 0; i < nums1.size(); i++) {
		if (map.find(nums1[i]) == map.end()) {
			map[nums1[i]] = i;
		}
	}

	for (int i = 0; i < nums2.size(); i++) {
		if (map.find(nums2[i]) != map.end()) {
			if (map[nums2[i]] == 1) {
				set.push_back(nums2[i]);
				map[nums2[i]]++;
			}
		}
	}
	return set;
}


string reverseVowels(string s) {
	if (s == "")return s;
	int phead = 0;
	int pfoot = s.size()-1;
	string vowels = "aeiouAEIOU";
	while (phead < pfoot) {
		while (vowels.find(s[phead]) == -1) {
			if(phead < pfoot)phead++;
			else break;
		}
		while (vowels.find(s[pfoot]) == -1)
		{
			if (phead < pfoot)pfoot--;
			else break;
		}
		cout << s[phead] << " " << s[pfoot] << endl;
		if (phead < pfoot) {
			char tmp = s[phead];
			s[phead] = s[pfoot];
			s[pfoot] = tmp;
			phead++;
			pfoot--;
		}
	}

	return s;
}


bool isPerfectSquare(int num) {
	if (num < 2)return true;
	int sqrtNum = num/2;
	while(sqrtNum*sqrtNum > num) {
		sqrtNum--;
	}
	if (sqrtNum*sqrtNum == num)return true;
	else return false;
}


//平方问题牛顿法 x[k+1] = 1/2(x[k]+num/x[k])
bool isPerfectSquareS1(int num) {
	if (num < 2)return true;
	long sqrtNum = num / 2;
	while (sqrtNum*sqrtNum > num) {
		sqrtNum = (sqrtNum + num / sqrtNum) / 2;
	}

	return sqrtNum * sqrtNum == num;
}
//利用规律 4 = 1+3;9 = 1+3+5
bool isPerfectSquareS2(int num) {
	if (num < 2)return true;
	int deNum = 1;
	while (num>0) {
		num -= deNum;
		deNum += 2;
	}
	return num == 0;
}

int guess(int num) {
};

int guessNumber(int n) {
	if (guess(n) == 0)return n;
	long mid = n / 2;
	long low(0), high(n);
	int res = guess(mid);
	while (res != 0) {
		if (res == -1) {
			high = mid;
			mid = (low + mid) / 2;
		}
		else {
			low = mid;
			mid = (high - mid) / 2 + mid;
		};
		res = guess(mid);
	}
	return mid;
}

bool canConstruct(string ransomNote, string magazine) {
	int sizeMa = magazine.size();
	for (auto c : ransomNote) {
		if (magazine.find(c) != -1) {
			magazine.erase(magazine.find(c),1);
		}
	}

	return (magazine.size() - sizeMa) == ransomNote.size();
}

char findTheDifference(string s, string t) {
	unordered_map<char, int> map;
	for (auto c : s) {
		if (map.find(c) != map.end()) {
			map[c]++;
		}
		else map[c] = 1;
	}
	for (auto c : t) {
		if (map.find(c) != map.end()) {
			map[c]--;
		}
		else {
			return c;
		}
	}

	for (auto c : map) {
		if (c.second == -1)return c.first;
	}
	return ' ';
}


// 该题数组快于map
char findTheDifferenceS1(string s, string t) {
	int mapS[26] = { 0 };
	int mapT[26] = { 0 };
	for (auto c : s) {
		mapS[c - 'a' - 1]++;
	}
	for (auto c : t) {
		mapT[c - 'a' - 1]++;
	}

	for (int i = 0; i < 26; i++) {
		if (mapS[i] != mapT[i])return i + 'a' + 1;
	}
	return ' ';
}

// n&(n-1)能够计算n中1的个数
int count(int hour){
	int tmp(hour);
	int count(0);
	while (tmp != 0){
		tmp = tmp&(tmp - 1);
		count++;
	}
	return count;
}
// 先判断小时 在半段分钟
vector<string> readBinaryWatch(int num) {
	vector<string> res;

	for (int i = 0; i < 12; i++) {
		if (count(i) == num) {
			res.push_back(to_string(i) + ":00");
		}
		else if(count(i)<num){
			for (int j = 0; j < 60; j++) {
				if ((count(i) + count(j)) == num) {
					res.push_back(to_string(i) + ":" + (j < 10 ? "0" + to_string(j) : to_string(j)));
				}
			}
		}
	}
	return res;
}

bool isSubsequence(string s, string t) {
	auto sptr = s.begin();
	auto tptr = t.begin();

	while (sptr != s.end()) {
		while (tptr!=t.end()&&*sptr != *tptr) {
			tptr++;
		}
		if (tptr == t.end())return false;
		sptr++;
		tptr++;
	}
	return true;
}


int curTree(TreeNode* node, bool isLeft) {
	if (node == NULL)return 0;
	if (node->left == NULL && node->right == NULL) {
		return isLeft ? node->val : 0;
	}
	return curTree(node->left, true) + curTree(node->right, false);
}

int sumOfLeftLeaves(TreeNode* root) {
	if (root == NULL || (root->left == NULL && root->right == NULL))return 0;
	return curTree(root->left, true) + curTree(root->right, false);
}

string toHex(int num) {
	int bHex;
	string hex = "";
	if (num == 0)return "0";
	while (num != 0 && num != -1) {
		bHex = num & 15;
		hex = char(bHex < 10 ? bHex + '0' : (bHex - 10) + 'a') + hex;
		num = num >> 4;
	}

	hex = num == 0 ? hex : string(8 - hex.size(), 'f') + hex;
	return hex;
}

int longestPalindrome(string s) {
	unordered_map<char, int> smap;
	bool Odd = false;
	int maxLen = 0;
	for (auto c : s) {
		if (smap.find(c) != smap.end())smap[c]++;
		else smap[c] = 1;
	}
	for (auto c : smap) {
		if (c.second % 2 == 0) {
			maxLen += c.second;
		}
		else {
			maxLen += c.second - 1;
			Odd = true;
		}
	}

	return maxLen + (Odd ? 1 : 0);
}


// set 有序唯一！！！ set的iterator 没有+，-，+=，-=的函数重载
int thirdMax(vector<int>& nums) {
	set<int> s(nums.begin(),nums.end());
	auto it = s.end();
	it--;
	if (s.size() >= 3) {
		it--;
		it--;
	}
	return *it;
}