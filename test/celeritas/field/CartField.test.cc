//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/CartField.test.cc
//---------------------------------------------------------------------------//

#include "corecel/data/HyperslabIndexer.hh"
#include "corecel/io/Logger.hh"
#include "celeritas/field/CartMapField.hh"
#include "celeritas/field/CartMapFieldInput.hh"
#include "celeritas/field/CartMapFieldParams.hh"

#include "celeritas_test.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// TESTS
//---------------------------------------------------------------------------//
using CartMapFieldTest = ::celeritas::test::Test;
TEST_F(CartMapFieldTest, all)
{
    CartMapFieldInput inp;
    inp.min_x = -2750;
    inp.max_x = 2750;
    inp.num_x = inp.max_x * 2 / 100;
    inp.min_y = -2750;
    inp.max_y = 2750;
    inp.num_y = inp.max_y * 2 / 100;
    inp.min_z = -6350;
    inp.max_z = 6350;
    inp.num_z = inp.max_z * 2 / 100;
    Array<size_type, 4> const dims{inp.num_x,
                                   inp.num_y,
                                   inp.num_z,
                                   static_cast<size_type>(CartAxis::size_)};
    size_type const total_points = inp.num_x * inp.num_y * inp.num_z;

    // Resize each component of the field
    inp.field.resize(static_cast<size_type>(CartAxis::size_) * total_points);

    // Fill with a simple field pattern
    HyperslabIndexer const flat_index{dims};
    for (size_type x = 0; x < inp.num_x; ++x)
    {
        for (size_type y = 0; y < inp.num_y; ++y)
        {
            for (size_type z = 0; z < inp.num_z; ++z)
            {
                auto arr = inp.field.begin() + flat_index(x, y, z, 0);
                arr[static_cast<size_type>(CartAxis::x)] = 0.1 * x;
                arr[static_cast<size_type>(CartAxis::y)] = 0.2 * y;
                arr[static_cast<size_type>(CartAxis::z)] = 0.3 * z;
            }
        }
    }

    CELER_LOG(info) << "field contains " << inp.field.size()
                    << " values, with dimensions: " << inp.num_x << " x "
                    << inp.num_y << " x " << inp.num_z;

    CartMapFieldParams field_map{inp};

    CartMapField calc_field(field_map.host_ref());

    // Sample the field

    // Define samples in cylindrical coordinates
    size_type const nx_samples = 3;
    size_type const ny_samples = 3;
    size_type const nz_samples = 3;
    std::vector<real_type> actual;

    for (size_type ix = 0; ix < nx_samples; ++ix)
    {
        real_type x = inp.min_x
                      + ix * (inp.max_x - inp.min_x) / (nx_samples - 1);
        for (size_type iy = 0; iy < ny_samples; ++iy)
        {
            real_type y = inp.min_y
                          + iy * (inp.max_y - inp.min_y) / (ny_samples - 1);
            for (size_type iz = 0; iz < nz_samples; ++iz)
            {
                real_type z = inp.min_z
                              + iz * (inp.max_z - inp.min_z) / (nz_samples - 1);

                CELER_LOG(info) << "x: " << x << " y: " << y << " z: " << z;
                Real3 field = calc_field({x, y, z});
                for (real_type f : field)
                {
                    actual.push_back(f);
                }
            }
        }
    }
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
