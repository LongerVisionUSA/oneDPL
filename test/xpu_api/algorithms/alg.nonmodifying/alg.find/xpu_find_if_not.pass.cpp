//===-- xpu_find_if_not.pass.cpp ------------------------------------------===//
//
// Copyright (C) Intel Corporation
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// This file incorporates work covered by the following copyright and permission
// notice:
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
//
//===----------------------------------------------------------------------===//

#include <oneapi/dpl/algorithm>

#include "support/test_iterators.h"

#include <cassert>
#include <CL/sycl.hpp>

struct ne
{
    ne(int val) : v(val) {}
    bool
    operator()(int v2) const
    {
        return v != v2;
    }
    int v;
};

template <class Iter>
void
test(sycl::queue& deviceQueue)
{
    bool ret = true;
    sycl::range<1> item1{1};
    {
        sycl::buffer<bool, 1> buffer1(&ret, item1);
        deviceQueue.submit([&](sycl::handler& cgh) {
            auto ret_acc = buffer1.get_access<sycl::access::mode::write>(cgh);
            cgh.single_task<Iter>([=]() {
                int ia[] = {0, 1, 2, 3, 4, 5};
                const unsigned s = sizeof(ia) / sizeof(ia[0]);
                auto r = dpl::find_if_not(Iter(ia), Iter(ia + s), ne(3));
                ret_acc[0] &= (*r == 3);
                r = dpl::find_if_not(Iter(ia), Iter(ia + s), ne(10));
                ret_acc[0] &= (r == Iter(ia + s));
            });
        });
    }
}

int
main()
{
    sycl::queue deviceQueue;
    test<input_iterator<const int*>>(deviceQueue);
    test<forward_iterator<const int*>>(deviceQueue);
    test<bidirectional_iterator<const int*>>(deviceQueue);
    test<random_access_iterator<const int*>>(deviceQueue);
    test<const int*>(deviceQueue);
    return 0;
}
