// -*- C++ -*-
//===----------------------------------------------------------------------===//
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

#include <oneapi/dpl/execution>
#include <oneapi/dpl/algorithm>

#include "support/test_config.h"

#if _ENABLE_RANGES_TESTING
#include <oneapi/dpl/ranges>
#endif

#include "support/utils.h"

#include <iostream>

int32_t
main()
{
#if _ENABLE_RANGES_TESTING
    constexpr int max_n = 10;
    int data[max_n]     = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    int expected[max_n] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    int rotate_val = 6;

    using namespace oneapi::dpl::experimental;

    auto view = nano::ranges::views::all(data) | ranges::views::rotate(rotate_val);
    {
        sycl::buffer<int> A(expected, expected + max_n);
        A.set_final_data(expected);
        A.set_write_back(true);

        ranges::copy(TestUtils::default_dpcpp_policy, view, A);
    }

    //check aasigment
    view[4] = -1;
    expected[4] = -1;

    ::std::cout << "view: ";
    for(auto v: view)
        ::std::cout << v << "  ";
    ::std::cout << ::std::endl;

    ::std::cout << "expected: ";
    for(auto v: expected)
        ::std::cout << v << "  ";
    ::std::cout << ::std::endl;

    ::std::cout << view.end() - view.begin() << ::std::endl;

    EXPECT_EQ_N(view.begin(), expected, max_n, "wrong result from rotate view on a device");

#endif //_ENABLE_RANGES_TESTING
    ::std::cout << TestUtils::done() << ::std::endl;
    return 0;
}
