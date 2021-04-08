
#include <CL/sycl.hpp>
#include <iostream>

int32_t
main()
{
    constexpr int max_n = 10;
    int data[max_n]     = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int expected[max_n] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

    {
        sycl::buffer<int> A(data, sycl::range<1>(max_n));
        sycl::buffer<int> B(expected, sycl::range<1>(max_n));

        sycl::queue myQueue;

        myQueue.submit([&](sycl::handler& cgh) {
           sycl::accessor in_acc { A, cgh, sycl::read_only };
           sycl::accessor out_acc { B, cgh, sycl::write_only };

           cgh.parallel_for<class Copy>(sycl::range</*dim=*/1>(max_n), [=](sycl::item</*dim=*/1> __item_id) {
           	auto __idx = __item_id.get_linear_id();
            		out_acc[__idx] = in_acc[__idx];
        	});
        });
    } 
    //A sync point on the end of the scope above - sycl::buffer desctruction.
    //But in sometimes we have got wrong "expected"

    ::std::cout << "data: ";
    for(auto v: data)
        ::std::cout << v << "  ";
    ::std::cout << ::std::endl;

    ::std::cout << "expected: ";
    for(auto v: expected)
        ::std::cout << v << "  ";
    ::std::cout << ::std::endl;

    assert(std::equal(data, data + max_n, expected, expected + max_n) && "test failed");

    return 0;
}
