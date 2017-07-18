/*!
 * @file InverseKinematicsData.cpp
 * @author Francesco Romano
 * @copyright 2016 iCub Facility - Istituto Italiano di Tecnologia
 *            Released under the terms of the LGPLv2.1 or later, see LGPL.TXT
 * @date 2016
 *
 */

#include "InverseKinematicsData.h"
#include "InverseKinematicsNLP.h"
#include "TransformConstraint.h"
#include <iDynTree/Core/Twist.h>
#include <iDynTree/Core/ClassicalAcc.h>
#include <iDynTree/Core/SpatialAcc.h>
#include <iDynTree/Model/Model.h>
#include <iDynTree/ModelIO/ModelLoader.h>
#include <iDynTree/Core/EigenHelpers.h>

#include <cassert>
#include <private/InverseKinematicsData.h>

namespace internal {
namespace kinematics {

    InverseKinematicsData::InverseKinematicsData(const InverseKinematicsData&) {}
    InverseKinematicsData& InverseKinematicsData::operator=(const InverseKinematicsData&) { return *this; }

    InverseKinematicsData::InverseKinematicsData()
    : m_defaultTargetResolutionMode(iDynTree::InverseKinematicsTreatTargetAsConstraintNone)
    , m_dofs(0)
    , m_rotationParametrization(iDynTree::InverseKinematicsRotationParametrizationQuaternion)
    , m_areBaseInitialConditionsSet(false)
    , m_areJointsInitialConditionsSet(false)
    , m_solver(NULL)
    // The default values for the ipopt related parameters are exactly the one of IPOPT,
    // see https://www.coin-or.org/Ipopt/documentation/node41.html
    //     https://www.coin-or.org/Ipopt/documentation/node42.html
    , m_maxIter(3000)
    , m_maxCpuTime(1e6)
    , m_tol(1e-8)
    , m_constrTol(1e-4)
    , m_verbosityLevel(0)
    {
        //These variables are touched only once.
        m_state.worldGravity.zero();

        m_state.baseTwist.zero();
        
        m_comTarget.isActive = false;
        m_comTarget.weight = 0;
        m_comTarget.desiredPosition.zero();
        m_comTarget.constraintTolerance = 1e-8;
        m_comTarget.isConstraint = false;
    }

    bool InverseKinematicsData::setModel(const iDynTree::Model& model, const std::vector<std::string> &consideredJoints)
    {
        // The inverse kinematics supports optimisation on a subset
        // of the joints of the full model.
        // Anyway, the joints configuration is necessary to properly update the
        // kinematics.
        // Because of this feature we have to separate the joints variable
        //m_ in two subset: q_opt and q_nopt.
        // The iDynTree KinDynComputations will be configured with all the joints
        // while all the optimisation-related methods will consider only q_opt

        // Save original model 
        m_originalModel = model;

        iDynTree::Model orderedModel(model);
        m_optimisedDofs = model.getNrOfDOFs();
        m_dofs = model.getNrOfDOFs();
        jointsMappingInfo.modelJointsToOptimisedMap.clear();
        jointsMappingInfo.optimisedToModelJointsMap.clear();


        if (!consideredJoints.empty()) {
            // Reorder the joints to have the optimised joints on top
            std::vector<std::string> orderedJoints = consideredJoints;
            for (iDynTree::JointIndex jointIdx = 0; jointIdx < model.getNrOfDOFs(); ++jointIdx) {
                std::string jointName = model.getJointName(jointIdx);
                std::vector<std::string>::const_iterator found = std::find(consideredJoints.begin(), consideredJoints.end(), jointName);

                size_t optimisedIndex;
                if (found == consideredJoints.end()) {
                    // joint not found => it should not be optimised. Add it to the end of the container
                    orderedJoints.push_back(jointName);
                    optimisedIndex = orderedJoints.size() - 1;
                } else {
                    optimisedIndex = std::distance(consideredJoints.begin(), found);
                }
                // now fill the map
                jointsMappingInfo.modelJointsToOptimisedMap.insert(IndicesMap::value_type(jointIdx, optimisedIndex));
                jointsMappingInfo.optimisedToModelJointsMap.insert(IndicesMap::value_type(optimisedIndex, jointIdx));
            }

            // Now we have partitioned the joints in two set: the first are the optimised joints.
            // The second, is composed of all the other joints
            iDynTree::ModelLoader loader;
            loader.loadReducedModelFromFullModel(model, orderedJoints);
            orderedModel = loader.model();
            m_optimisedDofs = consideredJoints.size();

        } else {
            // Optimised and model joints match
            // fill the map with a 1to1 association
            for (iDynTree::JointIndex jointIdx = 0; jointIdx < model.getNrOfDOFs(); ++jointIdx) {
                jointsMappingInfo.modelJointsToOptimisedMap.insert(IndicesMap::value_type(jointIdx, jointIdx));
                jointsMappingInfo.optimisedToModelJointsMap.insert(IndicesMap::value_type(jointIdx, jointIdx));
            }
        }
        assert(m_optimisedDofs <= orderedModel.getNrOfDOFs());


        bool result = m_dynamics.loadRobotModel(orderedModel);
        if (!result || !m_dynamics.isValid()) {
            std::cerr << "[ERROR] Error loading robot model" << std::endl;
            return false;
        }

        //prepare joint limits
        m_jointLimits.clear();
        m_jointLimits.resize(m_optimisedDofs);
        //TODO to be changed to +_ infinity
        //default: no limits
        m_jointLimits.assign(m_optimisedDofs, std::pair<double, double>(-2e+19, 2e+19));

        //for each joint, ask the limits
        //(this makes sense only for optimised joints)
        // As the optimised joints are the first one, we can iterate on those joints
        for (iDynTree::JointIndex jointIdx = 0; jointIdx < m_optimisedDofs; ++jointIdx) {
            iDynTree::IJointConstPtr joint = orderedModel.getJoint(jointIdx);
            //if the joint does not have limits skip it
            if (!joint->hasPosLimits())
                continue;
            //for each DoF modelled by the joint get the limits
            for (unsigned dof = 0; dof < joint->getNrOfDOFs(); ++dof) {
                if (!joint->getPosLimits(dof,
                                         m_jointLimits[joint->getDOFsOffset() + dof].first,
                                         m_jointLimits[joint->getDOFsOffset() + dof].second))
                    continue;
            }
        }

        //We set a new model, clear the variables
        clearProblem();

        updateRobotConfiguration();

        return true;
    }

    void InverseKinematicsData::clearProblem()
    {
        //resize vectors
        //OptimisationDofs size
        m_jointInitialConditions.resize(m_optimisedDofs);
        m_jointInitialConditions.zero();
        m_jointsResults.resize(m_optimisedDofs);
        m_jointsResults.zero();
        m_preferredJointsConfiguration.resize(m_optimisedDofs);
        m_preferredJointsConfiguration.zero();
        m_preferredJointsWeight = 1e-6;

        //Model dofs size
        m_state.jointsConfiguration.resize(m_dofs);
        m_state.jointsConfiguration.zero();

        m_state.jointsVelocity.resize(m_dofs);
        m_state.jointsVelocity.zero();

        m_state.basePose.setPosition(iDynTree::Position(0, 0, 0));
        m_state.basePose.setRotation(iDynTree::Rotation::Identity());

        m_constraints.clear();
        m_targets.clear();
        m_comHullConstraint.setActive(false);

        m_areBaseInitialConditionsSet = false;
        m_areJointsInitialConditionsSet = false;
        
        m_comTarget.isActive = false;
        m_comTarget.weight = 0;
        m_comTarget.desiredPosition.zero();
        m_comTarget.constraintTolerance = 1e-8;
    }

    bool InverseKinematicsData::addFrameConstraint(const kinematics::TransformConstraint& frameTransformConstraint)
    {
        int frameIndex = m_dynamics.getFrameIndex(frameTransformConstraint.getFrameName());
        if (frameIndex < 0)
            return false;

        //add the constraint to the set
        std::pair<TransformMap::iterator, bool> result = m_constraints.insert(TransformMap::value_type(frameIndex, frameTransformConstraint));
        return result.second;
    }

    bool InverseKinematicsData::addTarget(const kinematics::TransformConstraint& frameTransform)
    {
        int frameIndex = m_dynamics.getFrameIndex(frameTransform.getFrameName());
        if (frameIndex < 0)
            return false;

        std::pair<TransformMap::iterator, bool> result = m_targets.insert(TransformMap::value_type(frameIndex, frameTransform));
        // As the input is const, I can only modify it after insertion
        if (result.second) {
            result.first->second.setTargetResolutionMode(m_defaultTargetResolutionMode);
        }
        return result.second;
    }

    TransformMap::iterator InverseKinematicsData::getTargetRefIfItExists(const std::string targetFrameName)
    {
        // The error for this check is already printed in getFrameIndex
        int frameIndex = m_dynamics.getFrameIndex(targetFrameName);
        if (frameIndex == iDynTree::FRAME_INVALID_INDEX)
            return m_targets.end();

        // Find the target (if this fails, it will return m_targets.end()
        return m_targets.find(frameIndex);
    }

    void InverseKinematicsData::updatePositionTarget(TransformMap::iterator target, iDynTree::Position newPos, double newPosWeight)
    {
        assert(target != m_targets.end());
        target->second.setPosition(newPos);
        target->second.setPositionWeight(newPosWeight);
    }

    void InverseKinematicsData::updateRotationTarget(TransformMap::iterator target, iDynTree::Rotation newRot, double newRotWeight)
    {
        assert(target != m_targets.end());
        target->second.setRotation(newRot);
        target->second.setRotationWeight(newRotWeight);
    }

    iDynTree::KinDynComputations& InverseKinematicsData::dynamics() { return m_dynamics; }

    bool InverseKinematicsData::setInitialCondition(const iDynTree::Transform* baseTransform,
                                                    const iDynTree::VectorDynSize* initialJointCondition)
    {
        if (baseTransform) {
            m_baseInitialCondition = *baseTransform;
            m_areBaseInitialConditionsSet = true;
        }
        if (initialJointCondition) {
            assert(initialJointCondition->size() == m_jointInitialConditions.size());
            m_jointInitialConditions = *initialJointCondition;
            m_areJointsInitialConditionsSet = true;
        }

        return true;
    }

    bool InverseKinematicsData::setRobotConfiguration(const iDynTree::Transform& baseConfiguration,
                                                      const iDynTree::VectorDynSize& jointConfiguration)
    {
        assert(m_state.jointsConfiguration.size() == jointConfiguration.size());
        for (int index = 0; index < jointConfiguration.size(); ++index) {
            m_state.jointsConfiguration(jointsMappingInfo.modelJointsToOptimisedMap[index]) = jointConfiguration(index);
        }
        m_state.basePose = baseConfiguration;
        updateRobotConfiguration();
        return true;
    }

    bool InverseKinematicsData::setJointConfiguration(const std::string& jointName, const double jointConfiguration)
    {
        iDynTree::JointIndex jointIndex = m_dynamics.model().getJointIndex(jointName);
        if (jointIndex == iDynTree::JOINT_INVALID_INDEX) return false;
        m_state.jointsConfiguration(jointIndex) = jointConfiguration;
        updateRobotConfiguration();
        return true;
    }

    bool InverseKinematicsData::setDesiredJointConfiguration(const iDynTree::VectorDynSize& desiredJointConfiguration, const double weight)
    {
        assert(m_preferredJointsConfiguration.size() == desiredJointConfiguration.size());
        m_preferredJointsConfiguration = desiredJointConfiguration;
        if (weight >= 0.0) {
            m_preferredJointsWeight = weight;
        }
        return true;
    }

    void InverseKinematicsData::setRotationParametrization(enum iDynTree::InverseKinematicsRotationParametrization parametrization)
    {
        m_rotationParametrization = parametrization;
    }

    enum iDynTree::InverseKinematicsRotationParametrization InverseKinematicsData::rotationParametrization() { return m_rotationParametrization; }

    void InverseKinematicsData::updateRobotConfiguration()
    {
        m_dynamics.setRobotState(m_state.basePose,
                                 m_state.jointsConfiguration,
                                 m_state.baseTwist,
                                 m_state.jointsVelocity,
                                 m_state.worldGravity);
    }

    void InverseKinematicsData::prepareForOptimization()
    {
        //Do all stuff needed before starting an optimization problem
        //1) prepare initial condition if not explicitly set
        if (!m_areBaseInitialConditionsSet) {
            m_baseInitialCondition = m_state.basePose;
        }

        if (!m_areJointsInitialConditionsSet) {
            iDynTree::toEigen(m_jointInitialConditions) = iDynTree::toEigen(m_state.jointsConfiguration).head(m_optimisedDofs);
        }

        //2) Check joint limits..
        for (size_t i = 0; i < m_jointInitialConditions.size(); ++i) {
            //check joint to be inside limit
            double &jointValue = m_jointInitialConditions(i);
            if (jointValue < m_jointLimits[i].first || jointValue > m_jointLimits[i].second) {
                std::cerr << "[WARNING] InverseKinematics: joint with DOFIndex " << i << " initial condition is outside the limits " << m_jointLimits[i].first << " " << m_jointLimits[i].second << std::endl;
                //set the initial value to at the limit
                if (jointValue < m_jointLimits[i].first) {
                    jointValue = m_jointLimits[i].first;
                }

                if (jointValue > m_jointLimits[i].second) {
                    jointValue = m_jointLimits[i].second;
                }
            }
        }

    }

    void InverseKinematicsData::setDefaultTargetResolutionMode(iDynTree::InverseKinematicsTreatTargetAsConstraint mode)
    {
        m_defaultTargetResolutionMode = mode;
    }

    enum iDynTree::InverseKinematicsTreatTargetAsConstraint InverseKinematicsData::defaultTargetResolutionMode()
    {
        return m_defaultTargetResolutionMode;
    }

    void InverseKinematicsData::setTargetResolutionMode(TransformMap::iterator target, iDynTree::InverseKinematicsTreatTargetAsConstraint mode)
    {
       assert(target != m_targets.end());
       target->second.setTargetResolutionMode(mode);
    }

    enum iDynTree::InverseKinematicsTreatTargetAsConstraint InverseKinematicsData::targetResolutionMode(TransformMap::iterator target) const
    {
        assert(target != m_targets.end());
        return target->second.targetResolutionMode();
    }

    bool InverseKinematicsData::solveProblem()
    {
        Ipopt::ApplicationReturnStatus solverStatus;

        if (Ipopt::IsNull(m_solver)) {
            m_solver = IpoptApplicationFactory();

            //TODO: set options
            //For example, one needed option is the linear solver type
            //Best thing is to wrap the IPOPT options with new structure so as to abstract them
            m_solver->Options()->SetStringValue("hessian_approximation", "limited-memory");
            m_solver->Options()->SetIntegerValue("print_level",m_verbosityLevel);
            m_solver->Options()->SetIntegerValue("max_iter", m_maxIter);
            m_solver->Options()->SetNumericValue("max_cpu_time", m_maxCpuTime);
            m_solver->Options()->SetNumericValue("tol", m_tol);
            m_solver->Options()->SetNumericValue("constr_viol_tol", m_constrTol);
            m_solver->Options()->SetIntegerValue("acceptable_iter", 0);
#ifndef NDEBUG
            m_solver->Options()->SetStringValue("derivative_test", "first-order");
#endif
            if (!m_solverName.empty()) {
                m_solver->Options()->SetStringValue("linear_solver", m_solverName);
            } else {
                m_solver->Options()->GetStringValue("linear_solver", m_solverName, "");
            }

            solverStatus = m_solver->Initialize();
            if (solverStatus != Ipopt::Solve_Succeeded) {
                return false;
            }
        }

        prepareForOptimization();

        //instantiate the IpOpt problem
        internal::kinematics::InverseKinematicsNLP *iKin = new internal::kinematics::InverseKinematicsNLP(*this);
        //Do something (if necessary)
        Ipopt::SmartPtr<Ipopt::TNLP> problem(iKin);

        // Ask Ipopt to solve the problem
        solverStatus = m_solver->OptimizeTNLP(problem);

        if (solverStatus == Ipopt::Solve_Succeeded || solverStatus == Ipopt::Solved_To_Acceptable_Level ) {
            return true;
        } else {
            return false;
        }
    }

    void InverseKinematicsData::getSolution(iDynTree::Transform & baseTransformSolution,
                                            iDynTree::VectorDynSize & shapeSolution)
    {
        assert(shapeSolution.size() == m_optimisedDofs);
        baseTransformSolution = m_baseResults;
        shapeSolution         = m_jointsResults;
    }
    
    void InverseKinematicsData::setCoMTarget(iDynTree::Position& desiredPosition, double weight){
        this->m_comTarget.desiredPosition = desiredPosition;

        if (!this->m_comTarget.isActive
            && m_defaultTargetResolutionMode & iDynTree::InverseKinematicsTreatTargetAsConstraintPositionOnly) {
            this->m_comTarget.isConstraint = true;
        }
        
        if (!(weight < 0)) {
            this->m_comTarget.weight = weight;
        }

        this->m_comTarget.isActive = true;
    }

    void InverseKinematicsData::setCoMasConstraint(bool asConstraint)
    {
        this->m_comTarget.isConstraint = asConstraint;
    }

    bool InverseKinematicsData::isCoMaConstraint()
    {
        return this->m_comTarget.isConstraint;
    }

    void InverseKinematicsData::setCoMasConstraintTolerance(double TOL)
    {
        this->m_comTarget.constraintTolerance = TOL;
    }

    bool InverseKinematicsData::isCoMTargetActive(){
        return this->m_comTarget.isActive;
    }

    void InverseKinematicsData::setCoMTargetInactive()
    {
        this->m_comTarget.isActive = false;
        this->m_comTarget.weight = 0;
        this->m_comTarget.desiredPosition.zero();
    }

}
}
