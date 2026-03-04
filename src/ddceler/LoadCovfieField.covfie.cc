//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file ddceler/LoadCovfieField.covfie.cc
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
#include "celeritas/Units.hh"
#include "celeritas/field/CartMapFieldInput.hh"

namespace celeritas
{
namespace dd
{
namespace
{
//---------------------------------------------------------------------------//
/*!
 * Covfie field type as written by the covfie \c convert_bfield tool.
 *
 * Pipeline: affine → linear → strided → array(float3)
 *
 * The affine transform matrix has shape [3 × 4]:
 * \verbatim
 *   [sx   0   0  tx]
 *   [ 0  sy   0  ty]
 *   [ 0   0  sz  tz]
 * \endverbatim
 * mapping world coordinates to index space: idx_i = s_i * pos_i + t_i.
 * Inverting: pos_min_i = -t_i / s_i, pos_max_i = (n_i - 1 - t_i) / s_i.
 */
using storage_t = covfie::backend::array<covfie::vector::float3>;
using strided_t = covfie::backend::strided<covfie::vector::size3, storage_t>;
using interp_t = covfie::backend::linear<strided_t>;
using file_field_t = covfie::field<covfie::backend::affine<interp_t>>;

//---------------------------------------------------------------------------//
}  // namespace

//---------------------------------------------------------------------------//
/*!
 * Load a Cartesian magnetic field map from a binary covfie file.
 *
 * The file must have been written with the standard covfie \c convert_bfield
 * pipeline: affine → linear → strided → array(float3).
 *
 * Coordinates in the file are in centimetres and field values in tesla.
 * Both are converted to Celeritas native units on load.
 */
CartMapFieldInput LoadCovfieField(std::string const& filename)
{
    // Open the binary covfie file
    std::ifstream ifs(filename, std::ifstream::binary);
    CELER_VALIDATE(ifs.good(),
                   << "failed to open covfie field file '" << filename << "'");

    // Deserialise the field (must match the pipeline used when writing)
    file_field_t file_field(ifs);
    ifs.close();

    // Access the affine backend's owning data to read the transform
    auto const& affine_data = file_field.backend();
    auto const& mat = affine_data.get_configuration();

    // Access the strided backend to read the grid dimensions [nx, ny, nz]
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

    // Recover world-coordinate bounds (file units: centimetres)
    // and convert to Celeritas native units
    using namespace celeritas::units;
    auto const cm = static_cast<float>(centimeter);
    auto const to_native_field = static_cast<float>(tesla);

    CartMapFieldInput inp;

    inp.x.min = static_cast<real_type>((-tx / sx) * cm);
    inp.x.max = static_cast<real_type>(
        ((static_cast<float>(nx) - 1.f - tx) / sx) * cm);
    inp.x.num = static_cast<size_type>(nx);

    inp.y.min = static_cast<real_type>((-ty / sy) * cm);
    inp.y.max = static_cast<real_type>(
        ((static_cast<float>(ny) - 1.f - ty) / sy) * cm);
    inp.y.num = static_cast<size_type>(ny);

    inp.z.min = static_cast<real_type>((-tz / sz) * cm);
    inp.z.max = static_cast<real_type>(
        ((static_cast<float>(nz) - 1.f - tz) / sz) * cm);
    inp.z.num = static_cast<size_type>(nz);

    // Allocate field data: layout is [X][Y][Z][3]
    inp.field.resize(static_cast<size_type>(Axis::size_) * nx * ny * nz);

    // Access the strided backend directly to read grid node values without
    // going through the linear interpolator (which would try to access
    // out-of-bounds neighbors at the grid boundary)
    file_field_t::view_t field_view{file_field};
    // Chain: affine → linear → strided
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

                inp.field[base + 0] = static_cast<real_type>(bvec[0])
                                      * to_native_field;
                inp.field[base + 1] = static_cast<real_type>(bvec[1])
                                      * to_native_field;
                inp.field[base + 2] = static_cast<real_type>(bvec[2])
                                      * to_native_field;
            }
        }
    }

    CELER_ENSURE(inp);
    return inp;
}

//---------------------------------------------------------------------------//
}  // namespace dd
}  // namespace celeritas
