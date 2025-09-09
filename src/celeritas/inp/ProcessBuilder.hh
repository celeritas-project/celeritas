//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/inp/ProcessBuilder.hh
//! \todo Process building needs a total rewrite
//---------------------------------------------------------------------------//
#pragma once

#include <functional>
#include <memory>
#include <unordered_map>

#include "celeritas/io/ImportProcess.hh"

namespace celeritas
{
class ImportedProcesses;
class MaterialParams;
class ParticleParams;
class Process;

namespace inp
{
//---------------------------------------------------------------------------//
//! Input argument for user-provided process construction
struct ProcessBuilderInput
{
    using SPConstParticle = std::shared_ptr<ParticleParams const>;
    using SPConstMaterial = std::shared_ptr<MaterialParams const>;
    using SPConstImported = std::shared_ptr<ImportedProcesses const>;

    SPConstMaterial material;
    SPConstParticle particle;
    SPConstImported imported;
};

//!@{
//! \name User builder type aliases
using ProcessBuilderFunction
    = std::function<std::shared_ptr<Process>(ProcessBuilderInput const&)>;
using ProcessBuilderMap
    = std::unordered_map<ImportProcessClass, ProcessBuilderFunction>;
//!@}

//---------------------------------------------------------------------------//
}  // namespace inp
}  // namespace celeritas
