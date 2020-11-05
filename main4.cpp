#include <stdio.h>

int main()
{
    int a[5] = {1,2,3,4,5};
    int* p;
    p = a;
    if(p == a+0)
    {
        puts("p == a+0");
    }
    if(p+3 == a+3)
    {
        puts("p+3 == a+3");
    }
    printf("p = %x\n",p);
    printf(" a = %x\n",a);
    printf("a+3 = %x, p+3 = %x\n",a+3,p+3);
    printf("%d,%d,%d,%d,%d,%d\n",p[0],a[0],*p,*a,*&a[0],*&p[0]);
    printf("%d,%d,%d,%d,%d,%d\n",p[3],a[3],*(p+3),*(a+3),*&a[3],*&p[3]);


    return 0;
}
