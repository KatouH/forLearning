#include"DataStruct.h"
#include<string>
#include<vector>
#include<unordered_map>
#include<algorithm>
using namespace std;

bool isPalindrome(int x);//Accepted
int romanToInt(string s);//Accepted
string longestCommonPrefix(vector<string>& strs);
ListNode* mergeTwoLists(ListNode* l1, ListNode* l2);//Accepted
void deleteNode(ListNode* node);//Accepted
int maxSubArray(vector<int>& nums);
int climbStairs(int n);


int main() {
	cout << climbStairs(2) << endl;
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

string longestCommonPrefix(vector<string>& strs) {
	return "test";
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


class MinStack {
public:
	/** initialize your data structure here. */
	MinStack() {

	}

	void push(int x) {

	}

	void pop() {

	}

	int top() {

	}

	int getMin() {

	}
};

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