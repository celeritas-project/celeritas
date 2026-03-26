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

    std::size_t const nx = sizes[0];
    std::size_t const ny = sizes[1];
    std::size_t const nz = sizes[2];

    CELER_VALIDATE(nx > 1 && ny > 1 && nz > 1,
                   << "covfie field grid is degenerate: nx=" << nx
                   << " ny=" << ny << " nz=" << nz);

    // Extract scale (diagonal) and translation (last column) from the affine
    float const sx = mat(0, 0);
    float const sy = mat(1, 1);
    float const sz = mat(2, 2);
    float const tx = mat(0, 3);
    float const ty = mat(1, 3);
    float const tz = mat(2, 3);

    CELER_VALIDATE(sx > 0 && sy > 0 && sz > 0,
                   << "covfie affine transform has non-positive scale factors "
                   << "sx=" << sx << " sy=" << sy << " sz=" << sz);

    // Recover world-coordinate bounds: pos_min = -t/s, pos_max = (n-1-t)/s
    CartMapFieldInput inp;

    inp.x.min = static_cast<real_type>(-tx / sx);
    inp.x.max
        = static_cast<real_type>((static_cast<float>(nx) - 1.f - tx) / sx);
    inp.x.num = static_cast<size_type>(nx);

    inp.y.min = static_cast<real_type>(-ty / sy);
    inp.y.max
        = static_cast<real_type>((static_cast<float>(ny) - 1.f - ty) / sy);
    inp.y.num = static_cast<size_type>(ny);

    inp.z.min = static_cast<real_type>(-tz / sz);
    inp.z.max
        = static_cast<real_type>((static_cast<float>(nz) - 1.f - tz) / sz);
    inp.z.num = static_cast<size_type>(nz);

    // Allocate field data: layout is [X][Y][Z][3]
    inp.field.resize(static_cast<size_type>(Axis::size_) * nx * ny * nz);

    // Read grid node values directly from the strided backend
    file_cart_t::view_t field_view{file_field};
    auto const& strided_view = field_view.backend().get_backend().get_backend();

    for (auto ix : range(nx))
    {
        for (auto iy : range(ny))
        {
            for (auto iz : range(nz))
            {
                auto const bvec
                    = strided_view.at({static_cast<std::size_t>(ix),
                                       static_cast<std::size_t>(iy),
                                       static_cast<std::size_t>(iz)});

                size_type const base
                    = static_cast<size_type>((ix * ny + iy) * nz + iz)
                      * static_cast<size_type>(Axis::size_);

                inp.field[base + 0] = static_cast<real_type>(bvec[0]);
                inp.field[base + 1] = static_cast<real_type>(bvec[1]);
                inp.field[base + 2] = static_cast<real_type>(bvec[2]);
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

    std::size_t const nr = sizes[0];
    std::size_t const nz = sizes[1];

    CELER_VALIDATE(nr > 1 && nz > 1,
                   << "covfie BrBz field grid is degenerate: nr=" << nr
                   << " nz=" << nz);

    // Extract scale and translation from the [2x3] affine matrix
    float const sr = mat(0, 0);
    float const sz = mat(1, 1);
    float const tr = mat(0, 2);
    float const tz = mat(1, 2);

    CELER_VALIDATE(sr > 0 && sz > 0,
                   << "covfie BrBz affine transform has non-positive scale "
                   << "factors sr=" << sr << " sz=" << sz);

    // Recover world-coordinate bounds
    float const r_min = -tr / sr;
    float const r_max = (static_cast<float>(nr) - 1.f - tr) / sr;
    float const z_min = -tz / sz;
    float const z_max = (static_cast<float>(nz) - 1.f - tz) / sz;

    RZMapFieldInput inp;
    inp.num_grid_r = static_cast<unsigned int>(nr);
    inp.num_grid_z = static_cast<unsigned int>(nz);
    inp.min_r = static_cast<double>(r_min);
    inp.max_r = static_cast<double>(r_max);
    inp.min_z = static_cast<double>(z_min);
    inp.max_z = static_cast<double>(z_max);

    // Allocate field data: layout is [Z][R] (R has stride 1)
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
            auto const bvec = strided_view.at(
                {static_cast<std::size_t>(ir), static_cast<std::size_t>(iz)});

            // Index: [Z][R] with R stride 1
            std::size_t const idx = iz * nr + ir;

            inp.field_r[idx] = static_cast<double>(bvec[0]);
            inp.field_z[idx] = static_cast<double>(bvec[1]);
        }
    }

    CELER_LOG(debug) << "Loaded covfie BrBz field: " << nr << "x" << nz
                     << " grid, r=[" << r_min << ", " << r_max << "], z=["
                     << z_min << ", " << z_max << "]";
    CELER_ENSURE(inp);
    return inp;
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
