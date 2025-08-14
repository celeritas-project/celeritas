//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/InstancePathFinder.cc
//---------------------------------------------------------------------------//
#include "InstancePathFinder.hh"

#include "corecel/io/Join.hh"
#include "geocel/VolumeParams.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
/*!
 * Find volume instance IDs from a list of names.
 */
auto InstancePathFinder::operator()(IListSView names) const -> VecVolInst
{
    auto const& vol_inst = volumes_.volume_instance_labels();

    VecVolInst result;
    std::vector<std::string_view> missing;
    for (std::string_view sv : names)
    {
        auto lab = Label::from_separator(sv);
        auto vi = vol_inst.find_exact(lab);
        if (!vi)
        {
            missing.push_back(sv);
            continue;
        }
        result.push_back(vi);
    }
    CELER_VALIDATE(missing.empty(),
                   << "missing PVs from stack: "
                   << join(missing.begin(), missing.end(), ','));
    return result;
}

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
