interface A{
    void fun();
}
class B implements A{

   public void fun(){
        System.out.println("hello");
    }
}
class vaibhav {
    public static void main(String[] args) {
        B a1 = new B();
        a1.fun();      
    }
}