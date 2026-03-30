//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/LoadCovfieField.covfie.cc
//---------------------------------------------------------------------------//
#include "LoadCovfieField.hh"

#include <cstddef>
#include <fstream>
#include <string>
#include <covfie/core/backend/primitive/array.hpp>
#include <covfie/core/backend/transformer/affine.hpp>
#include <covfie/core/backend/transformer/linear.hpp>
#include <covfie/core/backend/transformer/strided.hpp>
#include <covfie/core/field.hpp>
#include <covfie/core/vector.hpp>

#include "corecel/Assert.hh"
#include "corecel/cont/Range.hh"
#include "corecel/io/Logger.hh"
#include "geocel/Types.hh"
#include "celeritas/Types.hh"

namespace celeritas
{
namespace
{
//---------------------------------------------------------------------------//
// Stateless covfie interpolation backend used solely for deserialization.
// Any stateless backend (linear, nearest_neighbour, etc.) produces an
// identical binary format because stateless backends write no IO headers.
// We use linear as an arbitrary choice; the file contents are the same
// regardless.
template<class B>
using deserialization_interp_t = covfie::backend::linear<B>;

// Covfie 3D field type for Cartesian maps.
using storage3_t = covfie::backend::array<covfie::vector::float3>;
using strided3_t = covfie::backend::strided<covfie::vector::size3, storage3_t>;
using file_cart_t
    = covfie::field<covfie::backend::affine<deserialization_interp_t<strided3_t>>>;

// Covfie 2D field type for BrBz cylindrical maps.
using storage2_t = covfie::backend::array<covfie::vector::float2>;
using strided2_t = covfie::backend::strided<covfie::vector::size2, storage2_t>;
using file_rz_t
    = covfie::field<covfie::backend::affine<deserialization_interp_t<strided2_t>>>;

//---------------------------------------------------------------------------//
/*!
 * Recover world-coordinate grid from a covfie affine axis.
 *
 * The affine transform maps world coordinates to grid indices:
 *   grid_index = scale * world_pos + translation
 * so world_pos_min = -t/s and world_pos_max = (n - 1 - t) / s.
 */
AxisGrid<real_type> from_affine(float scale, float translation, std::size_t n)
{
    CELER_VALIDATE(scale > 0,
                   << "covfie affine transform has non-positive scale factor "
                   << scale);
    CELER_VALIDATE(n > 1,
                   << "covfie field grid axis is degenerate with " << n
                   << " points");

    AxisGrid<real_type> result;
    result.min = -translation / scale;
    result.max = (static_cast<float>(n) - 1.f - translation) / scale;
    result.num = static_cast<size_type>(n);
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Load a Cartesian magnetic field map from a binary covfie file.
 */
CartMapFieldInput load_covfie_cart_field(std::string const& filename)
{
    std::ifstream ifs(filename, std::ifstream::binary);
    CELER_VALIDATE(ifs.good(),
                   << "failed to open covfie field file '" << filename << "'");

    file_cart_t file_field(ifs);
    ifs.close();

    // Access the affine backend for the transform matrix
    auto const& affine_data = file_field.backend();
    auto const& mat = affine_data.get_configuration();

    // Access the strided backend for grid dimensions [nx, ny, nz]
    auto const& strided_data = affine_data.get_backend().get_backend();
    auto const sizes = strided_data.get_configuration();

    // Recover world-coordinate grid from affine transform
    CartMapFieldInput inp;
    inp.x = from_affine(mat(0, 0), mat(0, 3), sizes[0]);
    inp.y = from_affine(mat(1, 1), mat(1, 3), sizes[1]);
    inp.z = from_affine(mat(2, 2), mat(2, 3), sizes[2]);

    // Allocate field data: layout is [X][Y][Z][3]
    std::size_t const nx = inp.x.num;
    std::size_t const ny = inp.y.num;
    std::size_t const nz = inp.z.num;
    std::size_t const num_components = static_cast<std::size_t>(Axis::size_);
    inp.field.resize(num_components * nx * ny * nz);

    // Read grid node values directly from the strided backend
    file_cart_t::view_t field_view{file_field};
    auto const& strided_view = field_view.backend().get_backend().get_backend();

    for (auto ix : range(nx))
    {
        for (auto iy : range(ny))
        {
            for (auto iz : range(nz))
            {
                auto const bvec = strided_view.at({ix, iy, iz});

                auto const base = ((ix * ny + iy) * nz + iz) * num_components;

                inp.field[base + 0] = bvec[0];
                inp.field[base + 1] = bvec[1];
                inp.field[base + 2] = bvec[2];
            }
        }
    }

    CELER_LOG(debug) << "Loaded covfie Cartesian field: " << nx << "x" << ny
                     << "x" << nz << " grid, x=[" << inp.x.min << ", "
                     << inp.x.max << "], y=[" << inp.y.min << ", " << inp.y.max
                     << "], z=[" << inp.z.min << ", " << inp.z.max << "]";
    CELER_ENSURE(inp);
    return inp;
}

//---------------------------------------------------------------------------//
/*!
 * Load a 2D BrBz cylindrical magnetic field map from a binary covfie file.
 */
RZMapFieldInput load_covfie_rz_field(std::string const& filename)
{
    std::ifstream ifs(filename, std::ifstream::binary);
    CELER_VALIDATE(ifs.good(),
                   << "failed to open covfie field file '" << filename << "'");

    file_rz_t file_field(ifs);
    ifs.close();

    // Access the affine backend for the [2x3] transform matrix
    auto const& affine_data = file_field.backend();
    auto const& mat = affine_data.get_configuration();

    // Access the strided backend for grid dimensions [nr, nz]
    auto const& strided_data = affine_data.get_backend().get_backend();
    auto const sizes = strided_data.get_configuration();

    // Recover world-coordinate grids from affine transform
    // Note: 2D affine has translation in column 2, not 3
    auto grid_r = from_affine(mat(0, 0), mat(0, 2), sizes[0]);
    auto grid_z = from_affine(mat(1, 1), mat(1, 2), sizes[1]);

    RZMapFieldInput inp;
    inp.num_grid_r = grid_r.num;
    inp.num_grid_z = grid_z.num;
    inp.min_r = grid_r.min;
    inp.max_r = grid_r.max;
    inp.min_z = grid_z.min;
    inp.max_z = grid_z.max;

    // Allocate field data: layout is [Z][R] (R has stride 1)
    std::size_t const nr = grid_r.num;
    std::size_t const nz = grid_z.num;
    std::size_t const grid_size = nr * nz;
    inp.field_r.resize(grid_size);
    inp.field_z.resize(grid_size);

    // Read grid node values directly from the strided backend
    file_rz_t::view_t field_view{file_field};
    auto const& strided_view = field_view.backend().get_backend().get_backend();

    for (auto iz : range(nz))
    {
        for (auto ir : range(nr))
        {
            auto const bvec = strided_view.at({ir, iz});

            // Index: [Z][R] with R stride 1
            std::size_t const idx = iz * nr + ir;

            inp.field_r[idx] = bvec[0];
            inp.field_z[idx] = bvec[1];
        }
    }

    CELER_LOG(debug) << "Loaded covfie BrBz field: " << nr << "x" << nz
                     << " grid, r=[" << grid_r.min << ", " << grid_r.max
                     << "], z=[" << grid_z.min << ", " << grid_z.max << "]";
    CELER_ENSURE(inp);
    return inp;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
