//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/field/detail/CovfieFieldType.hh
//---------------------------------------------------------------------------//
#pragma once

#include <covfie/core/backend/primitive/array.hpp>
#include <covfie/core/backend/transformer/affine.hpp>
#include <covfie/core/backend/transformer/linear.hpp>
#include <covfie/core/backend/transformer/strided.hpp>
#include <covfie/core/field.hpp>

#include "corecel/Config.hh"

#if CELERITAS_USE_CUDA
#    include <covfie/cuda/backend/primitive/cuda_texture.hpp>
#elif CELERITAS_USE_HIP
#    include <covfie/hip/backend/primitive/hip_device_array.hpp>
#endif

namespace celeritas
{
//---------------------------------------------------------------------------//
//! Covfie field type
#if CELERITAS_USE_CUDA

using covfie_field_d = covfie::field<covfie::backend::affine<
    covfie::backend::cuda_texture<covfie::vector::float3, covfie::vector::float3>>>;

#elif CELERITAS_USE_HIP

using covfie_field_d = covfie::field<covfie::backend::affine<
    covfie::backend::hip_device_array<covfie::vector::float3,
                                      covfie::vector::float3>>>;

#endif

using covfie_field = covfie::field<covfie::backend::affine<covfie::backend::linear<
    covfie::backend::strided<covfie::vector::size3,
                             covfie::backend::array<covfie::vector::float3>>>>>;
using covfie_field_d = covfie_field;

//---------------------------------------------------------------------------//
}  // namespace celeritas