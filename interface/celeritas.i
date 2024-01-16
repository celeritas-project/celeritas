//---------------------------------*-SWIG-*----------------------------------//
// Copyright 2020-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas.i
//---------------------------------------------------------------------------//
%module "celeritas"

%include <cstring.i>
%include <std_map.i>
%include <std_string.i>
%include <std_vector.i>

%include "detail/macros.i"
%feature("flatnested");

//---------------------------------------------------------------------------//
// CONFIG FILE
//---------------------------------------------------------------------------//
%{
#include "celeritas_version.h"
#include "celeritas_config.h"
#include "celeritas_cmake_strings.h"
%}

%include "celeritas_version.h"
%include "celeritas_config.h"
%include "celeritas_cmake_strings.h"

//---------------------------------------------------------------------------//
// ASSERTIONS
//---------------------------------------------------------------------------//

%{
#include <stdexcept>
%}

%include <exception.i>

%exception {
  try { $action }
  catch (const std::exception& e) {
    SWIG_exception(SWIG_RuntimeError, e.what());
  }
}

//---------------------------------------------------------------------------//
// CORECEL
//---------------------------------------------------------------------------//

%{
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
%}

%ignore celeritas::Byte;
%include "corecel/Macros.hh"
%include "corecel/Types.hh"

%template(VecReal) std::vector<celeritas::real_type>;

//---------------------------------------------------------------------------//
// CELERITAS
//---------------------------------------------------------------------------//

%{
#include "celeritas/Units.hh"
#include "celeritas/Constants.hh"
%}

%include "celeritas/Units.hh"
%include "celeritas/Constants.hh"

//---------------------------------------------------------------------------//

%{
#include "celeritas/phys/AtomicNumber.hh"
#include "celeritas/phys/PDGNumber.hh"
%}

%include "celeritas/phys/AtomicNumber.hh"
%include "celeritas/phys/PDGNumber.hh"

//---------------------------------------------------------------------------//
// IO
//---------------------------------------------------------------------------//

%{
#include "celeritas/io/ImportData.hh"
%}

namespace celeritas
{
%celer_rename_to_cstring(table_type, ImportTableType);
%celer_rename_to_cstring(units, ImportUnits);
%celer_rename_to_cstring(vector_type, ImportPhysicsVectorType);
%celer_rename_to_cstring(process_type, ImportProcessType);
%celer_rename_to_cstring(process_class, ImportProcessClass);
%celer_rename_to_cstring(model, ImportModelClass);
%rename(process_class_to_geant_name) to_geant_name(ImportProcessClass);
%rename(model_to_geant_name) to_geant_name(ImportModelClass);

%rename(xs_lo) ImportTableType::lambda;
%rename(xs_hi) ImportTableType::lambda_prim;
}

%include "celeritas/io/ImportParameters.hh"

%include "celeritas/io/ImportPhysicsVector.hh"
%template(VecImportPhysicsVector) std::vector<celeritas::ImportPhysicsVector>;

%include "celeritas/io/ImportPhysicsTable.hh"
%template(VecImportPhysicsTable) std::vector<celeritas::ImportPhysicsTable>;

%include "celeritas/io/ImportModel.hh"
%template(VecImportModel) std::vector<celeritas::ImportModel>;
%template(VecImportMscModel) std::vector<celeritas::ImportMscModel>;

%include "celeritas/io/ImportProcess.hh"
%template(VecImportProcess) std::vector<celeritas::ImportProcess>;

%include "celeritas/io/ImportParticle.hh"
%template(VecImportParticle) std::vector<celeritas::ImportParticle>;

%include "celeritas/io/ImportElement.hh"
%template(VecImportElement) std::vector<celeritas::ImportElement>;

%include "celeritas/io/ImportMaterial.hh"
%template(VecImportMaterial) std::vector<celeritas::ImportMaterial>;

%include "celeritas/io/ImportVolume.hh"
%template(VecImportVolume) std::vector<celeritas::ImportVolume>;

%include "celeritas/io/ImportAtomicRelaxation.hh"
%template(VecImportAtomicTransition) std::vector<celeritas::ImportAtomicTransition>;

%include "celeritas/io/ImportSBTable.hh"

%include "celeritas/io/ImportLivermorePE.hh"
%template(VecImportLivermoreSubshell) std::vector<celeritas::ImportLivermoreSubshell>;

%template(MapIAR) std::map<int, celeritas::ImportAtomicRelaxation>;
%template(MapLivermorePE) std::map<int, celeritas::ImportLivermorePE>;
%template(MapImportSB) std::map<int, celeritas::ImportSBTable>;
%include "celeritas/io/ImportData.hh"

//---------------------------------------------------------------------------//

%{
#include "celeritas/io/SeltzerBergerReader.hh"
%}

%include "celeritas/io/SeltzerBergerReader.hh"

//---------------------------------------------------------------------------//

%{
#include "celeritas/io/LivermorePEReader.hh"
%}

%include "celeritas/io/LivermorePEReader.hh"

//---------------------------------------------------------------------------//
// EXT
//---------------------------------------------------------------------------//
%{
#include "celeritas/ext/RootImporter.hh"
#include "celeritas/ext/RootExporter.hh"
#include "celeritas/ext/GeantPhysicsOptions.hh"
#include "celeritas/ext/GeantSetup.hh"
#include "celeritas/ext/GeantImporter.hh"
%}

namespace celeritas
{
%celer_rename_to_cstring(brems_model, BremsModelSelection);
%celer_rename_to_cstring(msc_model, MscModelSelection);
%celer_rename_to_cstring(relaxation_selection, RelaxationSelection);

%warnfilter(362) GeantSetup::operator=;
}

%include "celeritas/ext/RootImporter.hh"
%include "celeritas/ext/RootExporter.hh"
%include "celeritas/ext/GeantPhysicsOptions.hh"
%include "celeritas/ext/GeantSetup.hh"
%include "celeritas/ext/GeantImporter.hh"

// vim: set ft=lex ts=2 sw=2 sts=2 :
