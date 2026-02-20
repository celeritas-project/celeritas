//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/detail/OpticalStepParams.cc
//---------------------------------------------------------------------------//
#include "OpticalStepParams.hh"

namespace celeritas
{
namespace optical
{
namespace detail
{
using celeritas::make_aux_state;

//---------------------------------------------------------------------------//
OpticalStepParams::OpticalStepParams(std::string&& label, AuxId id)
    : label_(std::move(label)), aux_id_(id)
{
    HostVal<OpticalStepParamsData> host_data;
    data_ = ParamsDataStore{std::move(host_data)};
}
//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace optical
}  // namespace celeritas
