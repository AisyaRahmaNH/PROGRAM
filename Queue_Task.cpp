#include <iostream>
using namespace std;

// prepare a queue space
struct queueLL 
{
    // data / member
    int data;

    // pointer next
    queueLL *next;
};

int maxQueueLL = 25;
queueLL *head, *tail, *cur, *del, *newNode;

// count Linked List
int countLL() 
{
    if (head == NULL) 
    {
        return 0;
    } 
    else 
    {
        int many = 0;
        cur = head;
        while (cur != NULL) 
        {
            cur = cur->next;
            many++;
        }
        return many;
    };
}

// isFull Linked List
bool isFullLL() {
    if (countLL() == maxQueueLL) {
        return true;
    } else {
        return false;
    };
}

// isEmpty Linked List
bool isEmptyLL() {
    if (countLL() == 0) {
        return true;
    } else {
        return false;
    };
}

// Enqueue Linked List
void enqueueLL(int data) {
    if (isFullLL()) {
        cout << "The queue is full!" << endl;
    } else {
        if (isEmptyLL()) {
            head = new queueLL();
            head->data = data;
            head->next = NULL;
            tail = head;
        } else {
            newNode = new queueLL();
            newNode->data = data;
            newNode->next = NULL;
            tail->next = newNode;
            tail = newNode;
        };
    };
}

// Dequeue Linked List
void dequeueLL() {
    if (isEmptyLL()) {
        cout << "The queue is empty!" << endl;
    } else {
        del = head;
        head = head->next;
        del->next = NULL;
        delete del;
    };
}

// Display Linked List
void displayLL() {
    cout << "Queue Data : " << endl;
    if (isEmptyLL()) {
        cout << "The queue is empty!" << endl;
    } else {
        cout << "Quantity of queue data : " << countLL() << endl;
        cur = head;
        int number = 1;
        while (number <= maxQueueLL) {
            if (cur != NULL) {
                cout << number << ". " << cur->data << endl;
                cur = cur->next;
            } else {
                cout << number << ". " << "(Empty)" << endl;
            };

            number++;
        };
    };
    
    cout << "\n" << endl;
}

// Check The Size
int size() {
    int count = 0;
    queueLL* cur = head;

    while (cur != NULL) {
        count++;
        cur = cur->next;
    }

    return count;
}

int main() {
    srand(time(NULL));

    // Randomly insert 10 data to the queue
    for (int i = 0; i < 10; ++i) {
        int data = ((rand() % 1000) - 500) / 100.0;
        enqueueLL(data);
    };

    displayLL();
    dequeueLL();
    cout << "Size: " << size() << endl;
    cout << "Is Empty: " << (isEmptyLL() ? "Yes" : "No") << endl;

    return 0;
}
