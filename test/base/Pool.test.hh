//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Pool.test.hh
//---------------------------------------------------------------------------//

#include "base/Pool.hh"

namespace celeritas_test
{
using celeritas::Ownership;
using celeritas::MemSpace;

//---------------------------------------------------------------------------//
// MOCK POOLS
//---------------------------------------------------------------------------//

struct MockElement
{
    int            atomic_number = 0;
    double atomic_mass;
};

template<Ownership W, MemSpace M>
struct MockMaterial
{
    template<class T> using PItem = celeritas::PoolItem<T, W, M>;

    double     number_density;
    PItem<int> element_ids;
};


template<Ownership W, MemSpace M>
struct MockParamsPools
{
    template<class T> using Pool = celeritas::Pool<T, W, M>;

    Pool<MockElement>  elements;
    Pool<int>          element_ids;
    Pool<MockMaterial<W, M>> materials;
    int                max_element_components;
};

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Input data
struct PTestInput
{
    int num_threads;
};

//---------------------------------------------------------------------------//
//! Output results
struct PTestOutput
{
};

//---------------------------------------------------------------------------//
//! Run on device and return results
PTestOutput p_test(PTestInput);

//---------------------------------------------------------------------------//
} // namespace celeritas_test
