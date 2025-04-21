#ifndef ddceler_h
#define ddceler_h 1

#include <DDG4/Geant4PhysicsList.h>
#include <DDG4/Geant4RunAction.h>
#include <DDG4/Geant4EventAction.h>
#include <accel/SimpleOffload.hh>
#include <G4EmStandardPhysics.hh>

/// Namespace for the AIDA detector description toolkit
namespace dd4hep  {

  /// Namespace for the Geant4 based simulation part of the AIDA detector description toolkit
  namespace sim  {
    
    // Global shared setup options
    celeritas::SetupOptions& CelerSetupOptions();
    // Shared data and GPU setup
    celeritas::SharedParams& CelerSharedParams();
    // Thread-local transporter
    celeritas::LocalTransporter& CelerLocalTransporter();
    // Thread-local offload
    celeritas::SimpleOffload& CelerSimpleOffload();

    class EMPhysicsConstructor final : public G4EmStandardPhysics
    {
    public:
        using G4EmStandardPhysics::G4EmStandardPhysics;

        void ConstructProcess() override;
    };
    
    class CeleritasPhysicsListActionSequence :  public
    Geant4PhysicsListActionSequence {
    
    public:
       /// Standard constructor
      CeleritasPhysicsListActionSequence(Geant4Context* context, const std::string& nam);
      /// Default destructor
      virtual ~CeleritasPhysicsListActionSequence();
      
      G4VUserPhysicsList* activateCeleritas();
    }; 
   
    class CeleritasRunAction : public Geant4RunAction 
    {
      /// Standard constructor
      CeleritasRunAction(Geant4Context* context, const std::string& nam);
      /// Default destructor
      virtual ~CeleritasRunAction();
      /// Begin-of-run callback
      virtual void begin(const G4Run* run) override;
      /// End-of-run callback
      virtual void end(const G4Run* run) override;  
    };

    class CeleritasEventAction : public Geant4EventAction
    { 
      /// Standard constructor
      CeleritasEventAction(Geant4Context* context, const std::string& nam);
      /// Default destructor
      virtual ~CeleritasEventAction();
      /// Begin-of-run callback
      virtual void begin(const G4Event* event) override;
      /// End-of-run callback
      virtual void end(const G4Event* event) override;
    };

    class CeleritasInitialization : public Geant4Action
    {
      /// Standard constructor
      CeleritasInitialization(Geant4Context* context, const std::string& nam);
      /// Default destructor
      virtual ~CeleritasInitialization();
      /// Callback function to setup celeritas for the MT worker thread
      virtual void build() const;
      /// Callback function to setup celeritas for the MT master thread
      virtual void buildMaster() const;
    };

  } /* End namespace sim */  
} /* End namespace dd4hep*/

#endif
