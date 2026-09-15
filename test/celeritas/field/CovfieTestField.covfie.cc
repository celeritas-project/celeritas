//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/CovfieTestField.covfie.cc
//---------------------------------------------------------------------------//
#include "CovfieTestField.hh"

#include <fstream>
#include <covfie/core/algebra/affine.hpp>
#include <covfie/core/backend/primitive/array.hpp>
#include <covfie/core/backend/transformer/affine.hpp>
#include <covfie/core/backend/transformer/linear.hpp>
#include <covfie/core/backend/transformer/strided.hpp>
#include <covfie/core/field.hpp>
#include <covfie/core/parameter_pack.hpp>
#include <covfie/core/vector.hpp>

#include "corecel/Assert.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// Arbitrary stateless backend for serialization (see
// LoadCovfieField.covfie.cc)
template<class B>
using deserialization_interp_t = covfie::backend::linear<B>;

//---------------------------------------------------------------------------//
/*!
 * Write a tiny 3D covfie field (nx x ny x nz) to a file.
 *
 * Field values: Bx = ix, By = iy, Bz = iz (float).
 * Grid bounds: [0, nx-1] x [0, ny-1] x [0, nz-1] in native units.
 */
void write_cart_covfie(
    std::string const& path, std::size_t nx, std::size_t ny, std::size_t nz)
{
    using storage_t = covfie::backend::array<covfie::vector::float3>;
    using strided_t
        = covfie::backend::strided<covfie::vector::size3, storage_t>;
    using field_t = covfie::field<
        covfie::backend::affine<deserialization_interp_t<strided_t>>>;

    // Identity affine: translate=0, scale=1 (grid coords == world coords)
    auto translation = covfie::algebra::affine<3>::translation(0.f, 0.f, 0.f);
    auto scaling = covfie::algebra::affine<3>::scaling(1.f, 1.f, 1.f);

    field_t field(covfie::make_parameter_pack(
        field_t::backend_t::configuration_t(scaling * translation),
        field_t::backend_t::backend_t::configuration_t{},
        field_t::backend_t::backend_t::backend_t::configuration_t{nx, ny, nz}));

    // Fill: Bx=ix, By=iy, Bz=iz
    field_t::view_t fv(field);
    auto& strided_view = fv.backend().get_backend().get_backend();
    for (std::size_t ix = 0; ix < nx; ++ix)
    {
        for (std::size_t iy = 0; iy < ny; ++iy)
        {
            for (std::size_t iz = 0; iz < nz; ++iz)
            {
                auto& v = strided_view.at({ix, iy, iz});
                v[0] = static_cast<float>(ix);
                v[1] = static_cast<float>(iy);
                v[2] = static_cast<float>(iz);
            }
        }
    }

    std::ofstream ofs(path, std::ofstream::binary);
    CELER_ASSERT(ofs.good());
    field.dump(ofs);
}

//---------------------------------------------------------------------------//
/*!
 * Write a tiny 2D covfie field (nr x nz) to a file.
 *
 * Field values: Br = ir, Bz = iz (float).
 * Grid bounds: [0, nr-1] x [0, nz-1] in native units.
 */
void write_rz_covfie(std::string const& path, std::size_t nr, std::size_t nz)
{
    using storage_t = covfie::backend::array<covfie::vector::float2>;
    using strided_t
        = covfie::backend::strided<covfie::vector::size2, storage_t>;
    using field_t = covfie::field<
        covfie::backend::affine<deserialization_interp_t<strided_t>>>;

    auto translation = covfie::algebra::affine<2>::translation(0.f, 0.f);
    auto scaling = covfie::algebra::affine<2>::scaling(1.f, 1.f);

    field_t field(covfie::make_parameter_pack(
        field_t::backend_t::configuration_t(scaling * translation),
        field_t::backend_t::backend_t::configuration_t{},
        field_t::backend_t::backend_t::backend_t::configuration_t{nr, nz}));

    field_t::view_t fv(field);
    auto& strided_view = fv.backend().get_backend().get_backend();
    for (std::size_t ir = 0; ir < nr; ++ir)
    {
        for (std::size_t iz = 0; iz < nz; ++iz)
        {
            auto& v = strided_view.at({ir, iz});
            v[0] = static_cast<float>(ir);
            v[1] = static_cast<float>(iz);
        }
    }

    std::ofstream ofs(path, std::ofstream::binary);
    CELER_ASSERT(ofs.good());
    field.dump(ofs);
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
