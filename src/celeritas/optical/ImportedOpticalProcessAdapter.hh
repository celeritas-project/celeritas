//----------------------------------*-C++-*----------------------------------//
// Copyright 2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/ImportedOpticalProcessAdapter.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "celeritas/grid/GenericGridInserter.hh"
#include "celeritas/io/ImportData.hh"
#include "celeritas/io/ImportOpticalProcess.hh"
#include "celeritas/io/ImportPhysicsTable.hh"
#include "celeritas/mat/MaterialParams.hh"
#include "celeritas/optical/OpticalPhysics.hh"

namespace celeritas
{
class IOPAContextException : public RichContextException
{
  public:
    IOPAContextException(ImportOpticalProcessClass ipc, MaterialId mid);

    //! This class type
    char const* type() const final { return "ImportOpticalProcessAdapterContext"; }

    //! Save context to a JSON object
    void output(JsonPimpl*) const final {}

    //! Get an explanatory message
    char const* what() const noexcept final { return what_.c_str(); }

  private:
    std::string what_;
};

class ImportOpticalProcesses
{
  public:
    //!@{
    //! \name Type aliases
    using ImportOpticalProcessId = OpaqueId<ImportOpticalProcess>;
    using key_type = ImportOpticalProcessClass;
    //!@}

  public:
    // Construct with imported data
    static std::shared_ptr<ImportOpticalProcesses>
    from_import(ImportData const& data);

    // Construct with imported tables
    explicit ImportOpticalProcesses(std::vector<ImportOpticalProcess> io);

    // Return the process ID for the given process class
    ImportOpticalProcessId find(key_type) const;

    // Get the table for the given process ID
    inline ImportOpticalProcess const& get(ImportOpticalProcessId id) const;

    // Number of imported optical processes
    inline ImportOpticalProcessId::size_type size() const;

  private:
    std::vector<ImportOpticalProcess> processes_;
    std::map<key_type, ImportOpticalProcessId> ids_;
};

//---------------------------------------------------------------------------//
/*!
 *
 */
class ImportedOpticalProcessAdapter
{
  public:
    //!@{
    //! \name Type aliases
    using SPConstImported = std::shared_ptr<ImportOpticalProcesses const>;
    using OpticalGridInserter = GenericGridInserter<OpticalValueGridId>;
    //!@}

  public:
    //! Construct from shared table data
    ImportedOpticalProcessAdapter(SPConstImported imported,
                                  ImportOpticalProcessClass process_class);

    //! Construct step limits for the process
    std::vector<OpticalValueGridId>
    step_limits(OpticalGridInserter&, MaterialParams const&) const;

    //! Get the lambda table for the process
    inline ImportPhysicsTable const& get_lambda() const;

    //! Access the imported process
    inline ImportOpticalProcess const& process() const;

  private:
    SPConstImported imported_;
    ImportOpticalProcessClass process_class_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
ImportOpticalProcess const& ImportOpticalProcesses::get(ImportOpticalProcessId id) const
{
    CELER_EXPECT(id < this->size());
    return processes_[id.get()];
}

auto ImportOpticalProcesses::size() const -> ImportOpticalProcessId::size_type
{
    return processes_.size();
}

ImportPhysicsTable const& ImportedOpticalProcessAdapter::get_lambda() const
{
    return process().lambda_table;
}

ImportOpticalProcess const& ImportedOpticalProcessAdapter::process() const
{
    return imported_->get(imported_->find(process_class_));
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
